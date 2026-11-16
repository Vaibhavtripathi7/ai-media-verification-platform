import pg from "pg";
import { buildApp } from "./app.ts";
import { loadConfig } from "./config.ts";
import { DetectionStore } from "./db.ts";
import { InferenceClient } from "./inference.ts";

const config = loadConfig();
const pool = new pg.Pool({ connectionString: config.DATABASE_URL, max: 10 });
const store = new DetectionStore(pool);
const inference = new InferenceClient(config.INFERENCE_URL, config.INFERENCE_TIMEOUT_MS);

await store.migrate();
const app = await buildApp({ config, store, inference, model: await inference.model() });

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, async () => {
    app.log.info({ signal }, "shutting down");
    await app.close();
    await pool.end();
  });
}

await app.listen({ port: config.PORT, host: config.HOST });
