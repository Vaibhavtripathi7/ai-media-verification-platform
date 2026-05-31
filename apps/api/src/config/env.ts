import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const envSchema = z.object({
    NODE_ENV: z.enum([
        "developement",
        "test",
        "production"
    ]),

    PORT: z.coerce.number().default(3000),
    DATABASE_URL: z.string().min(1),
    REDIS_URL: z.string().min(1),
    QDRANT_URL: z.string().min(10)

});

const parsedEnv = envSchema.safeParse(process.env);
if (!parsedEnv.success) {
    console.error(
        "Invalide env variables:",
        parsedEnv.error.flatten().fieldErrors
    );
    process.exit(1);
}

export const env = parsedEnv.data;
