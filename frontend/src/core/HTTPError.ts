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

/**
 * Build a diagnostic message for an axios error caught during a file upload.
 * Distinguishes cancellation, network failure, HTTP error with FastAPI detail,
 * and file-too-large (413) so the user gets actionable feedback.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function formatUploadError(error: any, fileSizeBytes?: number): string {
  // axios cancellation (user clicked cancel / component unmounted)
  if (error?.name === 'CanceledError' || error?.code === 'ERR_CANCELED') {
    return 'Upload canceled.';
  }

  const sizeMB =
    typeof fileSizeBytes === 'number' ? ` (${(fileSizeBytes / (1024 * 1024)).toFixed(1)} MB)` : '';

  // HTTP response received: the server rejected the upload with a status
  if (error?.response) {
    const status = error.response.status;
    if (status === 413) {
      return `File too large${sizeMB}: server rejected with HTTP 413. Increase the body-size limit on your server/proxy.`;
    }
    const detail = formatApiError(error.response.data, '');
    return `Upload rejected by server (HTTP ${status})${detail ? `: ${detail}` : ''}.`;
  }

  // No response received: the request never completed (network/proxy/timeout/CORS)
  if (error?.code === 'ERR_NETWORK' || error?.message === 'Network Error') {
    return `Network error during upload${sizeMB}: the connection was interrupted before the server responded. Possible causes: server timeout, proxy/body-size limit, or the server crashed mid-upload.`;
  }
  if (error?.code === 'ECONNABORTED' || /timeout/i.test(error?.message ?? '')) {
    return `Upload timed out${sizeMB}. The file is large and the server or proxy closed the connection before it finished.`;
  }

  return `Upload failed${sizeMB}: ${formatApiError(error)}`;
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
