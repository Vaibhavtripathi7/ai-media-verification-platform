import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import type { FastifyInstance } from "fastify";
import type { Config } from "./config.ts";
import type { DetectionStore } from "./db.ts";
import { HttpError } from "./errors.ts";
import type { InferenceClient } from "./inference.ts";

export type Verdict = "likely_ai" | "uncertain" | "likely_real";

/** The detector's raw logit mapped to a calibrated probability that the image is AI-generated. */
export function calibrate(logit: number, config: Config): number {
  return 1 / (1 + Math.exp(-(config.CALIBRATION_SLOPE * logit + config.CALIBRATION_INTERCEPT)));
}

export function verdictFor(aiProbability: number, config: Config): Verdict {
  if (aiProbability >= config.AI_THRESHOLD) return "likely_ai";
  if (aiProbability <= config.REAL_THRESHOLD) return "likely_real";
  return "uncertain";
}

export interface Dependencies {
  config: Config;
  store: DetectionStore;
  inference: InferenceClient;
}

interface Scored {
  logit: number;
  cached: boolean;
  timing: string;
}

const ms = (start: number) => (performance.now() - start).toFixed(1);

export async function verifyRoutes(app: FastifyInstance, deps: Dependencies) {
  const { config, store, inference } = deps;
  const model = config.DETECTOR_MODEL;
  // Identical images uploaded at the same time share one detector call.
  const inFlight = new Map<string, Promise<Scored>>();

  async function score(sha256: Buffer, image: Buffer, contentType: string): Promise<Scored> {
    // The table is a cache. If Postgres is down we still answer, just without it.
    const lookupStart = performance.now();
    const cached = await store.find(sha256, model).catch((err: unknown) => {
      app.log.error({ err }, "cache lookup failed");
      return undefined;
    });
    const lookup = `cache;dur=${ms(lookupStart)}`;
    if (cached !== undefined) {
      return { logit: cached, cached: true, timing: lookup };
    }

    const result = await inference.detect(image, contentType);
    // The calibration only means something for the model it was fitted on.
    if (result.model !== model) {
      throw new HttpError(
        502,
        "detector_mismatch",
        `the detector runs ${result.model}, but the calibration is for ${model}`,
      );
    }
    await store.save(sha256, model, result.logit).catch((err: unknown) => {
      app.log.error({ err }, "cache write failed");
    });

    return {
      logit: result.logit,
      cached: false,
      timing: [
        lookup,
        `decode;dur=${result.decode_ms}`,
        `wait;dur=${result.wait_ms}`,
        `model;dur=${result.model_ms}`,
      ].join(", "),
    };
  }

  app.addContentTypeParser(
    /^image\//,
    { parseAs: "buffer", bodyLimit: config.MAX_IMAGE_BYTES },
    (_req, body, done) => done(null, body),
  );

  app.post(
    "/api/v1/verify",
    { config: { rateLimit: { max: config.RATE_LIMIT_PER_MINUTE, timeWindow: 60_000 } } },
    async (req, reply) => {
      const image = req.body;
      if (!Buffer.isBuffer(image)) {
        throw new HttpError(
          415,
          "unsupported_media_type",
          "send the raw image with an image/* type",
        );
      }
      if (image.length === 0) {
        throw new HttpError(400, "empty_body", "request body is empty");
      }

      const sha256 = createHash("sha256").update(image).digest();
      const key = sha256.toString("hex");

      let pending = inFlight.get(key);
      if (!pending) {
        pending = score(sha256, image, req.headers["content-type"]!).finally(() =>
          inFlight.delete(key),
        );
        inFlight.set(key, pending);
      }
      const result = await pending;

      const aiProbability = calibrate(result.logit, config);
      reply.header("server-timing", result.timing);
      return {
        sha256: key,
        verdict: verdictFor(aiProbability, config),
        aiProbability: Number(aiProbability.toFixed(4)),
        thresholds: { ai: config.AI_THRESHOLD, real: config.REAL_THRESHOLD },
        model,
        cached: result.cached,
      };
    },
  );
}
