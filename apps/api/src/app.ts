import Fastify, { type FastifyError } from "fastify";
import type { Config } from "./config.ts";
import type { DetectionStore } from "./db.ts";
import { HttpError } from "./errors.ts";
import type { InferenceClient } from "./inference.ts";
import { verifyRoutes } from "./verify.ts";

interface Dependencies {
  config: Config;
  store: DetectionStore;
  inference: InferenceClient;
  model: string;
}

const CODES: Record<number, string> = {
  400: "bad_request",
  413: "image_too_large",
  415: "unsupported_media_type",
};

export async function buildApp(deps: Dependencies) {
  const app = Fastify({ logger: { level: deps.config.LOG_LEVEL } });

  app.setErrorHandler((err: FastifyError | HttpError, req, reply) => {
    const status = err.statusCode ?? 500;
    if (status >= 500) req.log.error({ err }, "request failed");
    if (err instanceof HttpError && err.retryAfter) reply.header("retry-after", err.retryAfter);

    const code = err instanceof HttpError ? err.code : (CODES[status] ?? "internal_error");
    const message = status >= 500 && !(err instanceof HttpError) ? "internal error" : err.message;
    return reply.code(status).send({ error: { code, message } });
  });

  app.setNotFoundHandler((req, reply) =>
    reply.code(404).send({ error: { code: "not_found", message: `no route for ${req.url}` } }),
  );

  app.get("/health", async () => ({ status: "ok" }));
  app.get("/ready", async (_req, reply) => {
    try {
      return { status: "ready", model: await deps.inference.model() };
    } catch {
      return reply.code(503).send({ status: "detector unavailable" });
    }
  });

  await app.register(verifyRoutes, deps);
  return app;
}
