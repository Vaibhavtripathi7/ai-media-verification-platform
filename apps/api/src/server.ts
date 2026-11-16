import { setTimeout as sleep } from "node:timers/promises";
import pg from "pg";
import { buildApp } from "./app.ts";
import { loadConfig } from "./config.ts";
import { DetectionStore } from "./db.ts";
import { InferenceClient } from "./inference.ts";

const config = loadConfig();
const pool = new pg.Pool({ connectionString: config.DATABASE_URL, max: 10 });
const store = new DetectionStore(pool);
const inference = new InferenceClient(config.INFERENCE_URL, config.INFERENCE_TIMEOUT_MS);

const app = await buildApp({ config, store, inference });

// Postgres may still be starting, as on a fresh `docker compose up`. The detector
// is not waited for: cached answers work without it, and /ready reports it.
for (let delay = 500; ; delay = Math.min(delay * 2, 10_000)) {
  try {
    await store.migrate();
    break;
  } catch (err) {
    app.log.warn({ err }, "postgres is not reachable yet, retrying");
    await sleep(delay);
  }
}

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, async () => {
    app.log.info({ signal }, "shutting down");
    await app.close();
    await pool.end();
  });
}

await app.listen({ port: config.PORT, host: config.HOST });
