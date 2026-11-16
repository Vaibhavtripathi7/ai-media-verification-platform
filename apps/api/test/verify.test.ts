import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, beforeEach, test } from "node:test";
import pg from "pg";
import { buildApp } from "../src/app.ts";
import { loadConfig } from "../src/config.ts";
import { DetectionStore } from "../src/db.ts";
import { InferenceClient } from "../src/inference.ts";
import { calibrate, verdictFor } from "../src/verify.ts";

const DATABASE_URL =
  process.env.DATABASE_URL ?? "postgres://postgres:postgres@localhost:5432/verify_test";
const MODEL = "test/detector@abc1234";
const IMAGE = Buffer.from("pretend these are png bytes");
// Nothing listens on port 9 on a dev box or a CI runner.
const CLOSED_PORT_URL = "http://127.0.0.1:9";

type Responder = (res: ServerResponse) => void;

function detection(logit: number, delayMs = 0): Responder {
  return (res) =>
    setTimeout(() => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          logit,
          model: MODEL,
          decode_ms: 1,
          wait_ms: 0,
          model_ms: 5,
        }),
      );
    }, delayMs);
}

function failure(status: number, detail: string, headers: Record<string, string> = {}): Responder {
  return (res) => {
    res.writeHead(status, { "content-type": "application/json", ...headers });
    res.end(JSON.stringify({ detail }));
  };
}

let respond: Responder;
let detectCalls = 0;
const detector = createServer((req, res) => {
  req.resume();
  req.on("end", () => {
    if (req.url === "/health") return detection(0)(res);
    detectCalls += 1;
    respond(res);
  });
});

const pool = new pg.Pool({ connectionString: DATABASE_URL });
const store = new DetectionStore(pool);
let detectorUrl: string;

before(async () => {
  await new Promise<void>((resolve) => detector.listen(0, "127.0.0.1", resolve));
  detectorUrl = `http://127.0.0.1:${(detector.address() as AddressInfo).port}`;
  await store.migrate();
});

beforeEach(async () => {
  respond = detection(2);
  detectCalls = 0;
  await pool.query("TRUNCATE detections");
});

after(async () => {
  detector.close();
  await pool.end();
});

async function start(env: Record<string, string> = {}, appStore: DetectionStore = store) {
  const config = loadConfig({
    DATABASE_URL,
    INFERENCE_URL: detectorUrl,
    INFERENCE_TIMEOUT_MS: "300",
    MAX_IMAGE_BYTES: "1000",
    CALIBRATION_SLOPE: "1",
    CALIBRATION_INTERCEPT: "0",
    AI_THRESHOLD: "0.8",
    REAL_THRESHOLD: "0.2",
    LOG_LEVEL: "silent",
    ...env,
  });
  const inference = new InferenceClient(config.INFERENCE_URL, config.INFERENCE_TIMEOUT_MS);
  return buildApp({ config, store: appStore, inference, model: MODEL });
}

type App = Awaited<ReturnType<typeof start>>;

function upload(app: App, body: Buffer | string = IMAGE, type = "image/png") {
  return app.inject({
    method: "POST",
    url: "/api/v1/verify",
    headers: { "content-type": type },
    body,
  });
}

test("scores a new image, then answers the same bytes from the cache", async () => {
  const app = await start();

  const first = await upload(app);
  assert.equal(first.statusCode, 200);
  assert.deepEqual(first.json(), {
    sha256: createHash("sha256").update(IMAGE).digest("hex"),
    verdict: "likely_ai",
    aiProbability: 0.8808,
    thresholds: { ai: 0.8, real: 0.2 },
    model: MODEL,
    cached: false,
  });
  assert.match(String(first.headers["server-timing"]), /^cache;dur=[\d.]+, decode;dur=1, /);

  const second = await upload(app);
  assert.equal(second.json().cached, true);
  assert.equal(second.json().aiProbability, 0.8808);
  assert.match(String(second.headers["server-timing"]), /^cache;dur=[\d.]+$/);
  assert.equal(detectCalls, 1);
});

test("identical uploads in flight at the same time share one detector call", async () => {
  respond = detection(0, 100);
  const app = await start();

  const results = await Promise.all(Array.from({ length: 5 }, () => upload(app)));

  assert.deepEqual(
    results.map((r) => r.statusCode),
    [200, 200, 200, 200, 200],
  );
  assert.equal(detectCalls, 1);
});

test("maps the probability onto the configured bands", () => {
  const config = loadConfig({ DATABASE_URL, AI_THRESHOLD: "0.8", REAL_THRESHOLD: "0.2" });
  assert.equal(verdictFor(0.8, config), "likely_ai");
  assert.equal(verdictFor(0.79, config), "uncertain");
  assert.equal(verdictFor(0.21, config), "uncertain");
  assert.equal(verdictFor(0.2, config), "likely_real");
});

test("calibrates the raw logit with the configured slope and intercept", () => {
  const config = loadConfig({ DATABASE_URL, CALIBRATION_SLOPE: "0.5", CALIBRATION_INTERCEPT: "1" });
  assert.equal(calibrate(-2, config), 0.5);
  assert.ok(calibrate(10, config) > calibrate(9, config));
});

test("refuses thresholds that overlap", () => {
  assert.throws(
    () => loadConfig({ DATABASE_URL, AI_THRESHOLD: "0.3", REAL_THRESHOLD: "0.6" }),
    /REAL_THRESHOLD/,
  );
});

test("only accepts raw image bytes", async () => {
  const app = await start();

  const json = await upload(
    app,
    JSON.stringify({ url: "https://example.com/a.png" }),
    "application/json",
  );
  assert.equal(json.statusCode, 415);
  assert.equal(json.json().error.code, "unsupported_media_type");

  assert.equal((await upload(app, IMAGE, "application/octet-stream")).statusCode, 415);
  assert.equal((await upload(app, Buffer.alloc(0))).statusCode, 400);
  assert.equal(detectCalls, 0);
});

test("rejects images over the size limit before calling the detector", async () => {
  const app = await start();
  const res = await upload(app, Buffer.alloc(1001, 1));
  assert.equal(res.statusCode, 413);
  assert.equal(res.json().error.code, "image_too_large");
  assert.equal(detectCalls, 0);
});

test("passes on the detector's answer for an unreadable image", async () => {
  respond = failure(422, "not a decodable image");
  const app = await start();
  const res = await upload(app);
  assert.equal(res.statusCode, 422);
  assert.deepEqual(res.json(), {
    error: { code: "invalid_image", message: "not a decodable image" },
  });
});

test("tells the client to retry when the detector is busy", async () => {
  respond = failure(503, "detector is busy", { "retry-after": "1" });
  const app = await start();
  const res = await upload(app);
  assert.equal(res.statusCode, 503);
  assert.equal(res.headers["retry-after"], "1");
  assert.equal(res.json().error.code, "detector_busy");
});

test("answers 503 when the detector is down and 504 when it is too slow", async () => {
  const down = await start({ INFERENCE_URL: CLOSED_PORT_URL });
  const unreachable = await upload(down);
  assert.equal(unreachable.statusCode, 503);
  assert.equal(unreachable.json().error.code, "detector_unavailable");

  respond = detection(0, 1000);
  const slow = await start();
  const timedOut = await upload(slow);
  assert.equal(timedOut.statusCode, 504);
  assert.equal(timedOut.json().error.code, "detector_timeout");
});

test("keeps answering when the cache database is unreachable", async () => {
  const deadPool = new pg.Pool({
    connectionString: "postgres://postgres:postgres@127.0.0.1:9/none",
    connectionTimeoutMillis: 200,
  });
  try {
    const app = await start({}, new DetectionStore(deadPool));
    const res = await upload(app);
    assert.equal(res.statusCode, 200);
    assert.equal(res.json().cached, false);
  } finally {
    await deadPool.end();
  }
});

test("rate limits each client", async () => {
  const app = await start({ RATE_LIMIT_PER_MINUTE: "2" });
  await upload(app);
  await upload(app);
  const third = await upload(app);
  assert.equal(third.statusCode, 429);
  assert.equal(third.json().error.code, "rate_limited");
});

test("is ready only while the detector answers", async () => {
  const up = await start();
  assert.deepEqual((await up.inject("/ready")).json(), { status: "ready", model: MODEL });

  const down = await start({ INFERENCE_URL: CLOSED_PORT_URL });
  assert.equal((await down.inject("/ready")).statusCode, 503);
});
