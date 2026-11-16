import Fastify from "fastify";
import type { Config } from "./config.ts";

export async function buildApp(config: Config) {
  const app = Fastify({ logger: { level: config.LOG_LEVEL } });
  app.get("/health", async () => ({ status: "ok" }));
  return app;
}
