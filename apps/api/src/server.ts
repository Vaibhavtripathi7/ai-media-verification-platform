import { createApp } from "./app";

async function startServer() {
  const app = createApp();

  try {
    await app.listen({
      port: 3000,
      host: "0.0.0.0"
    });

    console.log("API running on port 3000");
  } catch (error) {
    app.log.error(error);
    process.exit(1);
  }
}

startServer();