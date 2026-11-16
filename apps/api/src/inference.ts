import { HttpError } from "./errors.ts";

export interface Detection {
  logit: number;
  model: string;
  decode_ms: number;
  wait_ms: number;
  model_ms: number;
}

export class InferenceClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(baseUrl: string, timeoutMs: number) {
    this.baseUrl = baseUrl;
    this.timeoutMs = timeoutMs;
  }

  async model(): Promise<string> {
    const res = await fetch(new URL("/health", this.baseUrl), {
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) throw new Error(`inference health check returned ${res.status}`);
    const body = (await res.json()) as { model: string };
    return body.model;
  }

  async detect(image: Buffer, contentType: string): Promise<Detection> {
    let res: Response;
    let body: string;
    try {
      res = await fetch(new URL("/v1/detect", this.baseUrl), {
        method: "POST",
        headers: { "content-type": contentType },
        body: image,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      // The timeout also covers the body, so a detector that stalls after the
      // headers has to be caught here too.
      body = await res.text();
    } catch (err) {
      if (err instanceof DOMException && err.name === "TimeoutError") {
        throw new HttpError(504, "detector_timeout", "the detector took too long to answer");
      }
      throw new HttpError(503, "detector_unavailable", "the detector is not reachable", {
        cause: err,
      });
    }

    if (res.ok) return JSON.parse(body) as Detection;

    let detail: string | undefined;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail;
    } catch {
      // Not FastAPI's JSON error shape; the status code alone decides.
    }
    switch (res.status) {
      case 413:
        throw new HttpError(413, "image_too_large", detail ?? "image is too large");
      case 422:
        throw new HttpError(422, "invalid_image", detail ?? "not a decodable image");
      case 503:
        throw new HttpError(503, "detector_busy", "the detector is busy, try again shortly", {
          retryAfter: res.headers.get("retry-after") ?? "1",
        });
      default:
        throw new HttpError(502, "detector_error", `the detector returned ${res.status}`);
    }
  }
}
