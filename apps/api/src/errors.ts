export class HttpError extends Error {
  readonly statusCode: number;
  readonly code: string;
  readonly retryAfter?: string;

  constructor(
    statusCode: number,
    code: string,
    message: string,
    options: { retryAfter?: string; cause?: unknown } = {},
  ) {
    super(message, { cause: options.cause });
    this.statusCode = statusCode;
    this.code = code;
    this.retryAfter = options.retryAfter;
  }
}
