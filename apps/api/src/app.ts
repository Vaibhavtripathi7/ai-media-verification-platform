import Fastify from "fastify";

export function createApp() {
  const app = Fastify({
    logger: true
  });

  return app;
}