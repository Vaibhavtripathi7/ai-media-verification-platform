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
      .default(20 * 1024 * 1024),
    RATE_LIMIT_PER_MINUTE: z.coerce.number().int().positive().default(30),
    // Platt scaling and verdict bands, fitted for DETECTOR_MODEL on the validation
    // split by apps/inference/eval/report.py. See docs/evaluation.md.
    DETECTOR_MODEL: z
      .string()
      .default("buildborderless/CommunityForensics-DeepfakeDet-ViT@ac6ee45"),
    CALIBRATION_SLOPE: z.coerce.number().default(0.4065),
    CALIBRATION_INTERCEPT: z.coerce.number().default(1.5623),
    AI_THRESHOLD: z.coerce.number().min(0).max(1).default(0.541),
    REAL_THRESHOLD: z.coerce.number().min(0).max(1).default(0.212),
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
