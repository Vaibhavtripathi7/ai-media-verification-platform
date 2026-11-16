// Latency of a running API, measured from the client side.
//
//   node scripts/bench.ts photo.jpg render.png --requests 60
//
// Start the API with a high RATE_LIMIT_PER_MINUTE first, or most requests come back 429.
// Fresh-image runs append a counter after the end of the file: decoders ignore trailing
// bytes, so the pixels stay the same while the hash, and therefore the cache key, changes.

import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    url: { type: "string", default: "http://localhost:3000" },
    requests: { type: "string", default: "40" },
  },
});

const TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

if (positionals.length === 0) {
  console.error("usage: node scripts/bench.ts <image>... [--requests N] [--url URL]");
  process.exit(1);
}

const images = await Promise.all(
  positionals.map(async (path) => ({
    data: await readFile(path),
    type: TYPES[extname(path).toLowerCase()] ?? "application/octet-stream",
  })),
);
const total = Number(values.requests);
const endpoint = new URL("/api/v1/verify", values.url);
let counter = 0;

async function post(body: Buffer, type: string): Promise<{ ms: number; ok: boolean }> {
  const start = performance.now();
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": type },
      body: new Uint8Array(body),
    });
    await res.arrayBuffer();
    return { ms: performance.now() - start, ok: res.ok };
  } catch {
    return { ms: performance.now() - start, ok: false };
  }
}

async function run(name: string, concurrency: number, fresh: boolean) {
  const latencies: number[] = [];
  let failed = 0;
  let next = 0;
  const started = performance.now();

  async function worker() {
    while (next < total) {
      const image = images[next++ % images.length];
      const body = fresh
        ? Buffer.concat([image.data, Buffer.from(`\n${process.pid}-${counter++}`)])
        : image.data;
      const { ms, ok } = await post(body, image.type);
      if (ok) latencies.push(ms);
      else failed += 1;
    }
  }

  await Promise.all(Array.from({ length: concurrency }, worker));
  const seconds = (performance.now() - started) / 1000;
  if (latencies.length === 0) {
    console.log(`${name.padEnd(28)}all ${failed} requests failed`);
    return;
  }
  latencies.sort((a, b) => a - b);
  const at = (q: number) =>
    latencies[Math.min(latencies.length - 1, Math.ceil(q * latencies.length) - 1)];

  console.log(
    [
      name.padEnd(28),
      `p50 ${at(0.5).toFixed(0)} ms`.padEnd(14),
      `p95 ${at(0.95).toFixed(0)} ms`.padEnd(14),
      `p99 ${at(0.99).toFixed(0)} ms`.padEnd(14),
      `${(latencies.length / seconds).toFixed(1)} req/s`.padEnd(14),
      failed ? `${failed} failed` : "",
    ].join(""),
  );
}

// Warm the connection pool and put the plain images in the cache.
for (const image of images) await post(image.data, image.type);

await run("fresh image, 1 at a time", 1, true);
await run("fresh images, 4 at a time", 4, true);
await run("repeat image, 1 at a time", 1, false);
await run("repeat image, 32 at a time", 32, false);
