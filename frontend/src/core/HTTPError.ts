/**
 * Extract a human-readable message from an API error response.
 * Handles FastAPI detail strings/arrays, {error: "..."} objects, and plain strings.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function formatApiError(err: any, fallback = 'An unknown error occurred'): string {
  if (!err) return fallback;
  if (typeof err === 'string') return err;
  if (typeof err === 'object') {
    if (typeof err.detail === 'string') return err.detail;
    if (Array.isArray(err.detail))
      return err.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ');
    if (typeof err.error === 'string') return err.error;
    if (typeof err.message === 'string') return err.message;
  }
  return JSON.stringify(err);
}

export class HttpError extends Error {
  status: Response['status'];

  constructor(status: Response['status'], message: string) {
    super(message);
    this.status = status;

    // Set the prototype explicitly.
    Object.setPrototypeOf(this, HttpError.prototype);
  }

  toString() {
    return `${this.status}: ${this.message}`;
  }
}
