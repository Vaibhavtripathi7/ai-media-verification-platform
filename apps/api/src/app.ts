import { fileURLToPath } from "node:url";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyError } from "fastify";
import { HttpError } from "./errors.ts";
import { type Dependencies, verifyRoutes } from "./verify.ts";

const CODES: Record<number, string> = {
  400: "bad_request",
  413: "image_too_large",
  415: "unsupported_media_type",
  429: "rate_limited",
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

  await app.register(helmet, {
    contentSecurityPolicy: {
      directives: {
        "img-src": ["'self'", "blob:", "data:"],
        // The page is served over plain http in local setups.
        "upgrade-insecure-requests": null,
      },
    },
  });
  await app.register(rateLimit, { global: false });
  await app.register(fastifyStatic, {
    root: fileURLToPath(new URL("../public", import.meta.url)),
  });

  app.get("/health", async () => ({ status: "ok" }));
  app.get("/ready", async (_req, reply) => {
    const expected = deps.config.DETECTOR_MODEL;
    const model = await deps.inference.model().catch(() => null);
    if (model === expected) return { status: "ready", model };
    return reply.code(503).send({ status: "not ready", detector: model, expected });
  });

  await app.register(verifyRoutes, deps);
  return app;
}
