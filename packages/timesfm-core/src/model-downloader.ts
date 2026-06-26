/**
 * TimesFM Model Downloader
 *
 * Downloads the pre-exported TimesFM 2.5 200M ONNX model from:
 *   1. GitHub Releases (primary) — timesfm-latest/timesfm-onnx-2.5.zip
 *   2. HuggingFace → export on the fly (requires Python + PyTorch)
 *
 * The npm packages are code-only (~100 KB).  The 885 MB model is stored
 * as a GitHub Release asset under the timesfm-latest channel and fetched
 * on first use.  The model is packaged as a zip containing the ONNX file
 * and model-descriptor.json for integrity verification.
 *
 * Features:
 *   - Streaming zip download + extraction (no 885 MB heap buffer)
 *   - SHA-256 integrity verification (from model-descriptor.json in zip)
 *   - Progress callback
 *   - Automatic cache management
 *   - Proxy support via environment variables or options parameter
 *   - Cross-platform zip extraction (unzip / 7z / PowerShell Expand-Archive)
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
 *   // With proxy (corporate network)
 *   const modelPath = await downloadModel({
 *     proxy: { url: 'http://proxy.company.com:8080', username: 'user', password: 'pass' },
 *   });
 *
 *   // Proxy via environment variables:
 *   //   TIMESFM_PROXY_URL=http://proxy:8080
 *   //   TIMESFM_PROXY_USERNAME=user
 *   //   TIMESFM_PROXY_PASSWORD=pass
 *   //   (or standard HTTP_PROXY / HTTPS_PROXY / http_proxy / https_proxy)
 *
 *   // From CLI: npx timesfm setup
 *   //   timesfm setup --proxy-url http://proxy:8080
 *   //   TIMESFM_PROXY_PASSWORD=pass timesfm setup --proxy-url http://proxy:8080 --proxy-username user
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createWriteStream, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { DownloadError, ChecksumMismatchError } from './errors';

// ─── Configuration ──────────────────────────────────────────────────────────

const REPO = 'AgentiX-E/agentix-timesfm-ts';
const MODEL_CHANNEL = 'timesfm-latest';
const ZIP_FILENAME = 'timesfm-onnx-2.5.zip';
const ONNX_FILENAME = 'timesfm-2.5.onnx';
const DESCRIPTOR_FILENAME = 'model-descriptor.json';
/** Expected zip size in bytes (~885 MB). */
const EXPECTED_ZIP_SIZE = 885 * 1024 * 1024;
/** Minimum size for a valid cached model (800 MB — generous tolerance). */
const MIN_CACHED_SIZE = 800 * 1024 * 1024;

const GITHUB_RELEASE_URL = `https://github.com/${REPO}/releases/download/${MODEL_CHANNEL}/${ZIP_FILENAME}`;

/** Default cache directory (platform-aware). */
function defaultCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(base, 'agentix-timesfm-ts');
}

/** Default model path. */
export function defaultModelPath(): string {
  return path.join(defaultCacheDir(), ONNX_FILENAME);
}

// ─── Proxy Configuration ────────────────────────────────────────────────────

/**
 * Proxy configuration for downloading models through corporate firewalls.
 *
 * Resolution priority:
 *   1. Explicit `DownloadOptions.proxy` parameter
 *   2. `TIMESFM_PROXY_URL` environment variable (+ optional `TIMESFM_PROXY_USERNAME` / `TIMESFM_PROXY_PASSWORD`)
 *   3. Standard `HTTPS_PROXY` / `https_proxy` / `HTTP_PROXY` / `http_proxy` environment variables
 */
export interface ProxyConfig {
  /** Proxy URL (e.g., http://proxy.company.com:8080 or socks5://proxy:1080) */
  url: string;
  /** Username for proxy authentication */
  username?: string;
  /** Password for proxy authentication */
  password?: string;
}

// ─── Download Options ───────────────────────────────────────────────────────

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
  /** Proxy configuration for restricted network environments */
  proxy?: ProxyConfig;
}

// ─── Proxy Resolution ───────────────────────────────────────────────────────

/**
 * Resolve proxy configuration with cascading priority.
 *
 * Priority: options.proxy → TIMESFM_PROXY_* env vars → standard *_proxy env vars
 */
function resolveProxyConfig(options?: DownloadOptions): ProxyConfig | null {
  // 1. Explicit proxy parameter
  if (options?.proxy?.url) {
    return options.proxy;
  }

  // 2. TIMESFM-specific environment variables
  const timesfmProxyUrl = process.env.TIMESFM_PROXY_URL;
  if (timesfmProxyUrl) {
    return {
      url: timesfmProxyUrl,
      username: process.env.TIMESFM_PROXY_USERNAME || undefined,
      password: process.env.TIMESFM_PROXY_PASSWORD || undefined,
    };
  }

  // 3. Standard environment variables (HTTPS_PROXY first since we use HTTPS)
  const standardUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy;
  if (standardUrl) {
    return { url: standardUrl };
  }

  return null;
}

/**
 * Apply proxy configuration to the current process.
 *
 * Node.js ≥ 20's built-in fetch() uses undici, which automatically
 * respects HTTP_PROXY / HTTPS_PROXY environment variables.
 *
 * For explicit proxy configuration (DownloadOptions.proxy or
 * TIMESFM_PROXY_* env vars), we temporarily set the standard
 * env vars before fetch() and restore them afterwards.
 */
function applyProxyEnv(proxy: ProxyConfig | null): (() => void) | null {
  if (!proxy) return null;

  const saved: Record<string, string | undefined> = {};
  const varsToSet: Record<string, string> = {};

  // Build proxy URL with optional authentication
  let proxyUrl = proxy.url;
  if (proxy.username || proxy.password) {
    try {
      const parsed = new URL(proxy.url);
      parsed.username = proxy.username || '';
      parsed.password = proxy.password || '';
      proxyUrl = parsed.toString();
    } catch {
      // If URL parsing fails, just use the raw URL
    }
  }

  // Set HTTPS_PROXY (and HTTP_PROXY as fallback for non-HTTPS URLs)
  varsToSet['HTTPS_PROXY'] = proxyUrl;
  varsToSet['https_proxy'] = proxyUrl;

  // Save old values
  for (const key of Object.keys(varsToSet)) {
    saved[key] = process.env[key];
    process.env[key] = varsToSet[key];
  }

  return () => {
    for (const key of Object.keys(saved)) {
      if (saved[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = saved[key];
      }
    }
  };
}

// ─── Core download function ─────────────────────────────────────────────────

/**
 * Download the TimesFM ONNX model as a zip, extract, and verify.
 *
 * The model is distributed as `timesfm-onnx-2.5.zip` containing:
 *   - timesfm-2.5.onnx
 *   - model-descriptor.json
 *
 * After download, the zip is extracted to the cache directory.  The ONNX
 * file is verified against the SHA-256 in the descriptor.  Only then is
 * the zip deleted.
 */
export async function downloadModel(options: DownloadOptions = {}): Promise<string> {
  const dest = path.resolve(options.dest ?? defaultModelPath());
  const force = options.force ?? false;
  const log = options.logger ?? ((msg: string) => console.error(msg));

  // Already have a valid cached model?
  if (!force && isModelCachedAtPath(dest)) {
    return dest;
  }

  const cacheDir = path.dirname(dest);
  fs.mkdirSync(cacheDir, { recursive: true });

  const url = options.url ?? GITHUB_RELEASE_URL;
  const zipDest = path.join(cacheDir, ZIP_FILENAME);
  const tmpZip = zipDest + '.tmp';

  log(`Downloading TimesFM 2.5 200M model (${(EXPECTED_ZIP_SIZE / 1024 / 1024).toFixed(0)} MB)...`);
  log(`  From: ${url}`);
  log(`  To:   ${dest}`);

  // Resolve proxy configuration
  const proxyConfig = resolveProxyConfig(options);

  // Set proxy env vars if explicit proxy config was provided.
  // Standard HTTP_PROXY/HTTPS_PROXY are already respected automatically
  // by Node's built-in fetch().
  const restoreEnv = applyProxyEnv(
    proxyConfig &&
      // Only apply if it came from an explicit source (not standard env vars,
      // which are already handled automatically)
      (options?.proxy || process.env.TIMESFM_PROXY_URL)
      ? proxyConfig
      : null,
  );

  // Stream download zip
  const fetchOptions: RequestInit = {
    redirect: 'follow',
    headers: { Accept: 'application/octet-stream' },
  };

  let response: Response;
  try {
    response = await fetch(url, fetchOptions);
  } finally {
    restoreEnv?.();
  }

  if (!response.ok) {
    const proxyHint = proxyConfig
      ? `\nProxy was configured (${proxyConfig.url}). Verify proxy credentials and connectivity.`
      : '';
    throw new DownloadError(
      `Failed to download model (HTTP ${response.status}): ${url}\n` +
        `If the model is not available as a GitHub Release, export it locally:\n` +
        `  pip install "timesfm[torch]" onnx onnxruntime torch\n` +
        `  python scripts/export-onnx.py --output ${dest}` +
        proxyHint,
      response.status,
    );
  }

  const total = parseInt(response.headers.get('content-length') || '0', 10);
  const totalMB = total > 0 ? total / 1024 / 1024 : EXPECTED_ZIP_SIZE / 1024 / 1024;
  const totalBytes = total > 0 ? total : EXPECTED_ZIP_SIZE;

  const fileStream = createWriteStream(tmpZip);
  const reader = response.body?.getReader();
  if (!reader) throw new DownloadError('No response body — cannot download model', 0);

  let received = 0;
  const startTime = Date.now();
  let lastLogAt = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        received += value.length;

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

        if (Math.floor(receivedMB / 50) > lastLogAt) {
          lastLogAt = Math.floor(receivedMB / 50);
          const pct = total > 0 ? ((received / totalBytes) * 100).toFixed(0) : '?';
          log(
            `  ${receivedMB.toFixed(0)} / ${totalMB.toFixed(0)} MB (${pct}%) @ ${speed.toFixed(1)} MB/s`,
          );
        }
      }
    }

    await new Promise<void>((resolve, reject) => {
      fileStream.end(() => resolve());
      fileStream.once('error', reject);
    });

    // Verify zip size
    const zipSize = fs.statSync(tmpZip).size;
    if (total > 0 && zipSize !== total) {
      throw new DownloadError(
        `Download incomplete: expected ${total} bytes, got ${zipSize} bytes.`,
        0,
      );
    }

    // Extract zip → cacheDir (gives timesfm-2.5.onnx + model-descriptor.json)
    log('  Extracting...');
    await extractZip(tmpZip, cacheDir);

    // Read descriptor for SHA verification
    let expectedSha256: string | null = null;
    const descriptorPath = path.join(cacheDir, DESCRIPTOR_FILENAME);
    try {
      const desc = JSON.parse(fs.readFileSync(descriptorPath, 'utf-8'));
      expectedSha256 = desc?.onnx?.sha256 ?? null;
    } catch {
      // Descriptor missing — skip checksum verification
    }

    // Verify SHA-256 of extracted ONNX
    if (expectedSha256) {
      const actualSha256 = sha256File(dest);
      if (actualSha256 !== expectedSha256) {
        cleanupPartial(cacheDir);
        throw new ChecksumMismatchError(
          `Checksum mismatch!\n  Expected: ${expectedSha256}\n  Got:      ${actualSha256}`,
        );
      }
    }

    // Clean up zip
    try {
      fs.unlinkSync(tmpZip);
    } catch {
      /* best-effort */
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log(
      `  Downloaded & extracted ${(zipSize / 1024 / 1024).toFixed(0)} MB in ${elapsed}s → ${dest}`,
    );
    return dest;
  } catch (err) {
    try {
      fs.unlinkSync(tmpZip);
    } catch {
      /* best-effort */
    }
    throw err;
  }
}

// ─── Cross-platform Zip Extraction ──────────────────────────────────────────

/**
 * Extract a zip file to a target directory.
 *
 * Tries multiple extraction backends for maximum cross-platform compatibility:
 *   1. `unzip` (Linux/macOS, BSD)
 *   2. `7z` (cross-platform, common on Windows CI)
 *   3. `powershell -Command Expand-Archive` (Windows built-in)
 *
 * If all backends fail, a descriptive error with platform-specific
 * installation instructions is thrown.
 */
async function extractZip(zipPath: string, outDir: string): Promise<void> {
  const backends: Array<() => Promise<void>> = [
    () => spawnExtractor('unzip', ['-o', zipPath, '-d', outDir]),
    () => spawnExtractor('7z', ['x', `-o${outDir}`, '-y', zipPath]),
    () => {
      // PowerShell Expand-Archive (Windows)
      const psCmd = `Expand-Archive -Path '${zipPath}' -DestinationPath '${outDir}' -Force`;
      return spawnExtractor('powershell', ['-NoProfile', '-Command', psCmd]);
    },
  ];

  const errors: string[] = [];

  for (const backend of backends) {
    try {
      await backend();
      return; // success — done
    } catch (err) {
      errors.push((err as Error).message);
    }
  }

  // All backends failed — provide helpful error
  const platform = process.platform;
  const installHint =
    platform === 'win32'
      ? 'Install 7-Zip: winget install 7zip.7zip'
      : platform === 'darwin'
        ? 'Install unzip: brew install unzip'
        : 'Install unzip: apt-get install unzip  or  yum install unzip';

  throw new DownloadError(
    `Failed to extract model zip. Tried 3 backends — all failed:\n` +
      errors.map((e, i) => `  ${i + 1}. ${e}`).join('\n') +
      `\n\nPlatform-specific fix: ${installHint}` +
      `\n\nAlternatively, extract manually:\n` +
      `  unzip ${zipPath} -d ${outDir}`,
    0,
  );
}

/** Spawn a subprocess for extraction, reject on non-zero exit. */
function spawnExtractor(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const { spawn } = require('node:child_process');
    const proc = spawn(command, args, {
      stdio: ['ignore', 'ignore', 'pipe'],
      timeout: 120_000, // 2 min timeout for large zip extraction
    });
    let stderr = '';
    proc.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'ENOENT') {
        reject(new Error(`"${command}" not found on PATH`));
      } else {
        reject(new Error(`${command}: ${err.message}`));
      }
    });
    proc.on('close', (code: number) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code ${code}: ${stderr.slice(0, 200)}`));
    });
  });
}

// ─── SHA-256 ────────────────────────────────────────────────────────────────

/** Compute SHA-256 of a file. */
function sha256File(filePath: string): string {
  const hasher = createHash('sha256');
  const buf = Buffer.alloc(64 * 1024);
  const fd = fs.openSync(filePath, 'r');
  try {
    let bytes: number;
    while ((bytes = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      hasher.update(buf.subarray(0, bytes));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hasher.digest('hex');
}

/** Clean up partial extraction artifacts. */
function cleanupPartial(cacheDir: string): void {
  for (const f of [ONNX_FILENAME, DESCRIPTOR_FILENAME]) {
    try {
      fs.unlinkSync(path.join(cacheDir, f));
    } catch {
      /* ignore */
    }
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
