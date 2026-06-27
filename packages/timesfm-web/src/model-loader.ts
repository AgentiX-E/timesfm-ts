/**
 * Browser model loader — downloads TimesFM ONNX models via fetch().
 *
 * Provides a browser-native equivalent of the Node.js model-downloader,
 * supporting progress callbacks, cancellation via AbortSignal,
 * and returning ArrayBuffer for onnxruntime-web.
 *
 * @module model-loader
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelLoaderOptions {
  /** Progress callback: received bytes, total bytes (may be 0 if unknown). */
  onProgress?: (received: number, total: number) => void;
  /** AbortSignal for cancellation. */
  signal?: AbortSignal;
}

export interface ModelLoadResult {
  /** The raw ONNX model data. */
  buffer: ArrayBuffer;
  /** Total size in bytes. */
  sizeBytes: number;
  /** Content-Type from the response. */
  contentType: string;
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/**
 * Download a TimesFM ONNX model from a URL.
 *
 * Uses the Fetch API with ReadableStream for progress tracking.
 * The returned ArrayBuffer can be passed directly to
 * `TimesFMWebInferenceEngine.load()`.
 *
 * @param url     URL of the ONNX model file.
 * @param options Progress callback and AbortSignal.
 * @returns The model as an ArrayBuffer with metadata.
 *
 * @example
 * ```typescript
 * import { loadModelFromUrl } from '@agentix-e/timesfm-web';
 *
 * const { buffer } = await loadModelFromUrl(
 *   'https://github.com/.../releases/download/timesfm-latest/timesfm-2.5.onnx',
 *   {
 *     onProgress: (received, total) => {
 *       console.log(`Downloaded ${(received / 1024 / 1024).toFixed(1)} MB`);
 *     },
 *   },
 * );
 *
 * // Pass buffer to the engine
 * engine.load(buffer);
 * ```
 */
export async function loadModelFromUrl(
  url: string,
  options: ModelLoaderOptions = {},
): Promise<ModelLoadResult> {
  const { onProgress, signal } = options;

  const response = await fetch(url, {
    signal,
    headers: { Accept: 'application/octet-stream, */*' },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to download model from ${url}: HTTP ${response.status} ${response.statusText}`,
    );
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
  const contentType = response.headers.get('content-type') || 'application/octet-stream';

  // If no progress callback and content-length is known, use simple arrayBuffer()
  if (!onProgress || !contentLength || !response.body) {
    const buffer = await response.arrayBuffer();
    return {
      buffer,
      sizeBytes: buffer.byteLength,
      contentType,
    };
  }

  // Stream with progress tracking
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    if (value) {
      chunks.push(value);
      received += value.length;
      onProgress(received, contentLength);
    }
  }

  // Combine chunks into a single ArrayBuffer
  const buffer = new ArrayBuffer(received);
  const view = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of chunks) {
    view.set(chunk, offset);
    offset += chunk.length;
  }

  return {
    buffer,
    sizeBytes: received,
    contentType,
  };
}

/**
 * Check if a model is available at a given URL (HEAD request).
 *
 * Useful for checking cache freshness before downloading.
 *
 * @param url URL to check.
 * @returns The Content-Length header value, or null if not accessible.
 */
export async function checkModelAvailability(url: string): Promise<number | null> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    if (response.ok) {
      const len = parseInt(response.headers.get('content-length') || '0', 10);
      return len > 0 ? len : null;
    }
    return null;
  } catch {
    return null;
  }
}
