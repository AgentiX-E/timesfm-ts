/**
 * TimesFM Model Downloader
 *
 * Downloads the pre-exported TimesFM 2.5 200M ONNX model from:
 *   1. GitHub Releases (primary) — {REPO}/releases/download/model-latest/timesfm-2.5.onnx
 *   2. HuggingFace → export on the fly (requires Python + PyTorch)
 *
 * The npm packages are code-only (~100 KB).  The 885 MB model is stored
 * as a GitHub Release asset under the model-latest channel and fetched on first use.
 *
 * The model descriptor (model-descriptor.json) is also fetched from the
 * model-latest channel to provide the expected SHA-256 checksum for
 * integrity verification of the downloaded ONNX file.
 *
 * Features:
 *   - Streaming download (no 885 MB heap buffer)
 *   - SHA-256 integrity verification (via model-descriptor.json from model-latest channel)
 *   - Progress callback
 *   - Automatic cache management
 *
 * Usage:
 *   // Auto-download to default path
 *   const modelPath = await downloadModel();
 *
 *   // Specify destination with progress
 *   const modelPath = await downloadModel({
 *     dest: './my-model.onnx',
 *     onProgress: (received, total) => console.log(`${received}/${total} MB`),
 *   });
 *
 *   // From CLI: npx timesfm setup
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createWriteStream, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

// ─── Configuration ──────────────────────────────────────────────────────────

const REPO = 'AgentiX-E/agentix-timesfm-ts';
const MODEL_CHANNEL = 'model-latest';
const MODEL_FILENAME = 'timesfm-2.5.onnx';
const DESCRIPTOR_FILENAME = 'model-descriptor.json';
/** Expected model size in bytes (885 MB). */
const EXPECTED_MODEL_SIZE = 885 * 1024 * 1024;
/** Minimum size for a valid cached model (800 MB — generous tolerance). */
const MIN_CACHED_SIZE = 800 * 1024 * 1024;

const GITHUB_RELEASE_URL = `https://github.com/${REPO}/releases/download/${MODEL_CHANNEL}/${MODEL_FILENAME}`;
const META_URL = `https://github.com/${REPO}/releases/download/${MODEL_CHANNEL}/${DESCRIPTOR_FILENAME}`;

/** Default cache directory (platform-aware). */
function defaultCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(base, 'agentix-timesfm-ts');
}

/** Default model path. */
export function defaultModelPath(): string {
  return path.join(defaultCacheDir(), MODEL_FILENAME);
}

// ─── Progress callback type ────────────────────────────────────────────────

export interface DownloadOptions {
  /** Target file path (default: ~/.cache/agentix-timesfm-ts/timesfm-2.5.onnx) */
  dest?: string;
  /** Force re-download even if file exists */
  force?: boolean;
  /** Progress callback: (receivedMB, totalMB, speedMBs) */
  onProgress?: (received: number, total: number, speed: number) => void;
  /** Alternative download URL (for mirrors) */
  url?: string;
  /** Custom logger (defaults to console.error) */
  logger?: (msg: string) => void;
}

// ─── Core download function ─────────────────────────────────────────────────

/**
 * Download the TimesFM ONNX model with streaming and integrity check.
 *
 * If `force` is false (default) and a valid cached file exists, returns immediately.
 * Otherwise, downloads from the configured URL, verifies the SHA-256 checksum,
 * and reports progress.
 */
export async function downloadModel(options: DownloadOptions = {}): Promise<string> {
  const dest = path.resolve(options.dest ?? defaultModelPath());
  const force = options.force ?? false;
  const log = options.logger ?? ((msg: string) => console.error(msg));

  // Already have a valid cached model?
  if (!force && isModelCachedAtPath(dest)) {
    return dest;
  }

  // Ensure directory
  fs.mkdirSync(path.dirname(dest), { recursive: true });

  // Fetch model descriptor from model-latest channel for SHA-256 verification
  let expectedSha256: string | null = null;
  try {
    const metaResp = await fetch(META_URL, { redirect: 'follow' });
    if (metaResp.ok) {
      const meta = (await metaResp.json()) as { onnx?: { sha256?: string } };
      expectedSha256 = meta?.onnx?.sha256 ?? null;
    }
  } catch {
    // Meta fetch failed — skip checksum verification, rely on size check only
  }

  const url = options.url ?? GITHUB_RELEASE_URL;
  log(
    `Downloading TimesFM 2.5 200M model (${(EXPECTED_MODEL_SIZE / 1024 / 1024).toFixed(0)} MB)...`,
  );
  log(`  From: ${url}`);
  log(`  To:   ${dest}`);

  // Stream to temporary file first (avoid corrupting cache on failure)
  const tmpDest = dest + '.tmp';
  const response = await fetch(url, {
    redirect: 'follow',
    headers: { Accept: 'application/octet-stream' },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to download model (HTTP ${response.status}): ${url}\n` +
        `If the model is not available as a GitHub Release, export it locally:\n` +
        `  pip install "timesfm[torch]" onnx onnxruntime torch\n` +
        `  python scripts/export-onnx.py --output ${dest}`,
    );
  }

  const total = parseInt(response.headers.get('content-length') || '0', 10);
  const totalMB = total > 0 ? total / 1024 / 1024 : EXPECTED_MODEL_SIZE / 1024 / 1024;
  const totalBytes = total > 0 ? total : EXPECTED_MODEL_SIZE;

  // Stream directly to file via pipeline (no 885 MB heap buffer)
  const fileStream = createWriteStream(tmpDest);
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body — cannot download model');
  }

  const hasher = createHash('sha256');
  let received = 0;
  const startTime = Date.now();
  let lastLogAt = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        hasher.update(value);
        received += value.length;

        // Write chunk directly — no buffer accumulation
        await new Promise<void>((resolve, reject) => {
          if (!fileStream.write(value)) {
            fileStream.once('drain', resolve);
          } else {
            resolve();
          }
          fileStream.once('error', reject);
        });

        const receivedMB = received / 1024 / 1024;
        const elapsed = (Date.now() - startTime) / 1000;
        const speed = elapsed > 0 ? receivedMB / elapsed : 0;

        if (options.onProgress) {
          options.onProgress(receivedMB, totalMB, speed);
        }

        // Log every 50 MB
        if (Math.floor(receivedMB / 50) > lastLogAt) {
          lastLogAt = Math.floor(receivedMB / 50);
          const pct = total > 0 ? ((received / totalBytes) * 100).toFixed(0) : '?';
          log(
            `  ${receivedMB.toFixed(0)} / ${totalMB.toFixed(0)} MB (${pct}%) @ ${speed.toFixed(1)} MB/s`,
          );
        }
      }
    }

    // Finalize write
    await new Promise<void>((resolve, reject) => {
      fileStream.end(() => resolve());
      fileStream.once('error', reject);
    });

    // Verify size
    const finalSize = fs.statSync(tmpDest).size;
    if (total > 0 && finalSize !== total) {
      throw new Error(`Download incomplete: expected ${total} bytes, got ${finalSize} bytes.`);
    }

    // Verify checksum
    const checksum = hasher.digest('hex');
    if (expectedSha256 && checksum !== expectedSha256) {
      try {
        fs.unlinkSync(tmpDest);
      } catch {
        /* best-effort */
      }
      throw new Error(`Checksum mismatch!\n  Expected: ${expectedSha256}\n  Got:      ${checksum}`);
    }

    // Atomic rename: only move to final path after successful verification
    fs.renameSync(tmpDest, dest);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log(`  Downloaded ${(finalSize / 1024 / 1024).toFixed(0)} MB in ${elapsed}s → ${dest}`);
    return dest;
  } catch (err) {
    // Clean up temp file on failure
    try {
      fs.unlinkSync(tmpDest);
    } catch {
      /* best-effort */
    }
    throw err;
  }
}

// ─── Cache helpers ──────────────────────────────────────────────────────────

/**
 * Check if a valid model exists at the given path.
 * Uses the expected model size (885 MB ± tolerance) rather than
 * a loose 100 MB heuristic.
 */
function isModelCachedAtPath(p: string): boolean {
  if (!existsSync(p)) return false;
  try {
    return fs.statSync(p).size >= MIN_CACHED_SIZE;
  } catch {
    return false;
  }
}

/**
 * Check if the default model is already cached.
 */
export function isModelCached(): boolean {
  return isModelCachedAtPath(defaultModelPath());
}

/**
 * Get the path to a cached model, or null.
 */
export function getCachedModelPath(): string | null {
  const p = defaultModelPath();
  return isModelCachedAtPath(p) ? p : null;
}
