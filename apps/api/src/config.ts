import { z } from "zod";

const schema = z
  .object({
    PORT: z.coerce.number().int().positive().default(3000),
    HOST: z.string().default("0.0.0.0"),
    LOG_LEVEL: z
      .enum(["silent", "fatal", "error", "warn", "info", "debug", "trace"])
      .default("info"),
    DATABASE_URL: z.url(),
    INFERENCE_URL: z.url().default("http://localhost:8000"),
    INFERENCE_TIMEOUT_MS: z.coerce.number().int().positive().default(15_000),
    MAX_IMAGE_BYTES: z.coerce
      .number()
      .int()
      .positive()
      .default(10 * 1024 * 1024),
    // Platt scaling and verdict bands, all fitted on the validation split by
    // apps/inference/eval/report.py. See docs/evaluation.md.
    CALIBRATION_SLOPE: z.coerce.number().default(1),
    CALIBRATION_INTERCEPT: z.coerce.number().default(0),
    AI_THRESHOLD: z.coerce.number().min(0).max(1).default(0.5),
    REAL_THRESHOLD: z.coerce.number().min(0).max(1).default(0.5),
  })
  .refine((c) => c.REAL_THRESHOLD <= c.AI_THRESHOLD, {
    message: "REAL_THRESHOLD must not be above AI_THRESHOLD",
  });

export type Config = z.infer<typeof schema>;

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const result = schema.safeParse(env);
  if (!result.success) {
    throw new Error(`invalid configuration\n${z.prettifyError(result.error)}`);
  }
  return result.data;
}
