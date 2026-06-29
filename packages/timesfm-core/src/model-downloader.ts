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
 *   //   # or for Docker/K8s secrets:
 *   //   TIMESFM_PROXY_PASSWORD_FILE=/run/secrets/proxy-password
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
import { spawn } from 'node:child_process';
import { DownloadError, ChecksumMismatchError, ProxyAuthError } from './errors';

// ─── Configuration ──────────────────────────────────────────────────────────

const REPO = 'AgentiX-E/agentix-timesfm-ts';
const DESCRIPTOR_FILENAME = 'model-descriptor.json';

/** Precision-specific download profile. */
interface PrecisionProfile {
  /** Suffix appended to ONNX filename, e.g. '' for fp32, '-int8' for int8. */
  readonly suffix: string;
  /** Zip filename for the given precision. */
  readonly zipFilename: string;
  /** ONNX filename for the given precision. */
  readonly onnxFilename: string;
  /** Expected zip size in bytes. */
  readonly expectedZipSize: number;
  /** Minimum size for a valid cached model. */
  readonly minCachedSize: number;
}

const PRECISION_PROFILES: Readonly<Record<string, PrecisionProfile>> = Object.freeze({
  fp32: {
    suffix: '',
    zipFilename: 'timesfm-onnx-2.5.zip',
    onnxFilename: 'timesfm-2.5.onnx',
    expectedZipSize: 885 * 1024 * 1024,
    minCachedSize: 800 * 1024 * 1024,
  },
  int8: {
    suffix: '-int8',
    zipFilename: 'timesfm-onnx-2.5-int8.zip',
    onnxFilename: 'timesfm-2.5-int8.onnx',
    expectedZipSize: 230 * 1024 * 1024,
    minCachedSize: 200 * 1024 * 1024,
  },
});

/** Default precision for download when none specified. */
const DOWNLOAD_DEFAULT_PRECISION = 'fp32';

function precisionProfile(precision: string = DOWNLOAD_DEFAULT_PRECISION): PrecisionProfile {
  return PRECISION_PROFILES[precision] ?? PRECISION_PROFILES[DOWNLOAD_DEFAULT_PRECISION];
}

function releaseUrl(precision: string): string {
  const profile = precisionProfile(precision);
  const channel = precision === 'fp32' ? 'timesfm-latest' : `timesfm-latest-${precision}`;
  return `https://github.com/${REPO}/releases/download/${channel}/${profile.zipFilename}`;
}

/** Default cache directory (platform-aware). */
function defaultCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(base, 'agentix-timesfm-ts');
}

/** Default model path. */
export function defaultModelPath(precision?: string): string {
  return path.join(defaultCacheDir(), precisionProfile(precision).onnxFilename);
}

/** Backward-compatible: check both FP32 and INT8 cache. */
export function getCachedModelPath(): string | null {
  for (const prec of ['fp32', 'int8'] as const) {
    const p = defaultModelPath(prec);
    if (isModelCachedAtPath(p, precisionProfile(prec).minCachedSize)) return p;
  }
  return null;
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
  /** Precision variant to download. Default 'fp32'. 'int8' for quantized models. */
  precision?: string;
}

// ─── Proxy Resolution ───────────────────────────────────────────────────────

/**
 * Resolve proxy configuration with cascading priority.
 *
 * Priority: options.proxy → TIMESFM_PROXY_* env vars → standard *_proxy env vars
 *
 * TIMESFM_PROXY_PASSWORD_FILE can be used instead of TIMESFM_PROXY_PASSWORD
 * for secret management in Docker/Kubernetes environments. The file is read
 * synchronously — its contents are trimmed and used as the password.
 */
function resolveProxyConfig(options?: DownloadOptions): ProxyConfig | null {
  // 1. Explicit proxy parameter
  if (options?.proxy?.url) {
    return options.proxy;
  }

  // 2. TIMESFM-specific environment variables
  const timesfmProxyUrl = process.env.TIMESFM_PROXY_URL;
  if (timesfmProxyUrl) {
    // Support TIMESFM_PROXY_PASSWORD_FILE for Docker/Kubernetes secrets
    let password = process.env.TIMESFM_PROXY_PASSWORD || undefined;
    if (!password) {
      const passwordFile = process.env.TIMESFM_PROXY_PASSWORD_FILE;
      if (passwordFile) {
        try {
          password = fs.readFileSync(passwordFile, 'utf-8').trim();
        } catch {
          // File not found or unreadable — leave password undefined
        }
      }
    }
    return {
      url: timesfmProxyUrl,
      username: process.env.TIMESFM_PROXY_USERNAME || undefined,
      password,
    };
  }

  // 3. Standard environment variables (HTTPS_PROXY first since we use HTTPS)
  //    Skip if NO_PROXY / no_proxy matches the target host.
  const noProxy = (process.env.NO_PROXY || process.env.no_proxy || '').toLowerCase();

  const standardUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy;

  // If NO_PROXY matches the GitHub releases domain, skip standard proxy
  if (standardUrl && noProxy) {
    const noProxyPatterns = noProxy.split(',').map((p) => p.trim());
    const isExcluded = noProxyPatterns.some(
      (p) => p === '*' || p === 'github.com' || p === '.github.com',
    );
    if (!isExcluded) {
      return { url: standardUrl };
    }
    return null;
  }

  if (standardUrl) {
    return { url: standardUrl };
  }

  return null;
}

/**
 * Fallback: apply proxy configuration via environment variables.
 *
 * Used when undici ProxyAgent is not available (older Node.js or
 * bundled environments where `import('undici')` fails).
 *
 * Returns a restore function that undoes the mutation — callers MUST
 * invoke it after the download completes to avoid leaking proxy
 * settings into subsequent fetch() calls within the same process.
 *
 * ⚠️ This is a legacy fallback path. Node.js ≥ 20 bundles undici
 * internally and should always use the ProxyAgent dispatcher path.
 * The env-based path is retained for environments where dynamic
 * import of undici fails (e.g., some bundlers, very old Node.js).
 *
 * @returns A zero-argument function that restores the original proxy
 *          environment variables.
 */
function applyProxyEnv(proxy: ProxyConfig): () => void {
  const saved: Record<string, string | undefined> = {
    HTTPS_PROXY: process.env.HTTPS_PROXY,
    https_proxy: process.env.https_proxy,
  };

  let proxyUrl = proxy.url;
  if (proxy.username || proxy.password) {
    try {
      const parsed = new URL(proxy.url);
      parsed.username = proxy.username || '';
      parsed.password = proxy.password || '';
      proxyUrl = parsed.toString();
    } catch {
      // If URL parsing fails, use the raw URL
    }
  }
  process.env.HTTPS_PROXY = proxyUrl;
  process.env.https_proxy = proxyUrl;

  return () => {
    if (saved.HTTPS_PROXY !== undefined) {
      process.env.HTTPS_PROXY = saved.HTTPS_PROXY;
    } else {
      delete process.env.HTTPS_PROXY;
    }
    if (saved.https_proxy !== undefined) {
      process.env.https_proxy = saved.https_proxy;
    } else {
      delete process.env.https_proxy;
    }
  };
}

/**
 * Apply proxy configuration to fetch options using Node.js undici dispatcher.
 *
 * Node.js ≥ 20's built-in fetch() uses undici internally.  We pass proxy
 * configuration via the `dispatcher` option instead of mutating global
 * environment variables, avoiding race conditions from concurrent downloads
 * and side effects on other fetch() calls in the same process.
 */
async function applyProxyToFetch(
  proxy: ProxyConfig | null,
): Promise<{ fetchOptions: Pick<RequestInit, 'dispatcher'>; restoreEnv?: () => void }> {
  if (!proxy) return { fetchOptions: {} };

  try {
    // Node.js ≥ 20 bundles undici internally. Use dynamic import for ESM compatibility.
    const undici = await import('undici');
    const ProxyAgent = undici.ProxyAgent;

    let proxyUrl = proxy.url;
    if (proxy.username || proxy.password) {
      try {
        const parsed = new URL(proxy.url);
        parsed.username = proxy.username || '';
        parsed.password = proxy.password || '';
        proxyUrl = parsed.toString();
      } catch {
        // If URL parsing fails, use the raw URL
      }
    }

    const dispatcher = new ProxyAgent({
      uri: proxyUrl,
      keepAliveTimeout: 10_000,
      keepAliveMaxTimeout: 30_000,
    });

    return { fetchOptions: { dispatcher: dispatcher as RequestInit['dispatcher'] } };
  } catch {
    // Fallback: undici ProxyAgent not available (e.g., bundled environment).
    // Use environment variables — Node ≥ 20 undici respects HTTPS_PROXY.
    // Returns a restore function so the caller can undo the env mutation after download.
    if (proxy) {
      const restoreEnv = applyProxyEnv(proxy);
      return { fetchOptions: {}, restoreEnv };
    }
    return { fetchOptions: {} };
  }
}

// ─── Core download function ─────────────────────────────────────────────────
// v8 ignore: the download + extraction path requires GitHub Releases (network)
// and is exercised by the model-release workflow's validate job.  Cache-only
// paths (resolution, proxy config parsing, SHA-256) are covered by unit tests.

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
  const prec = options.precision ?? DOWNLOAD_DEFAULT_PRECISION;
  const profile = precisionProfile(prec);
  const dest = path.resolve(options.dest ?? defaultModelPath(prec));
  const force = options.force ?? false;
  const log = options.logger ?? ((msg: string) => console.error(msg));

  // Already have a valid cached model?
  if (!force && isModelCachedAtPath(dest, profile.minCachedSize)) {
    return dest;
  }

  const cacheDir = path.dirname(dest);
  fs.mkdirSync(cacheDir, { recursive: true });

  const url = options.url ?? releaseUrl(prec);
  const zipDest = path.join(cacheDir, profile.zipFilename);
  const tmpZip = zipDest + '.tmp';

  log(
    `Downloading TimesFM 2.5 200M model (${(profile.expectedZipSize / 1024 / 1024).toFixed(0)} MB)...`,
  );
  log(`  From: ${url}`);
  log(`  To:   ${dest}`);

  // Resolve proxy configuration
  const proxyConfig = resolveProxyConfig(options);

  // Apply proxy to fetch via undici dispatcher (preferred) or env vars (fallback)
  const { fetchOptions: proxyFetchOpts, restoreEnv } = await applyProxyToFetch(
    proxyConfig ? proxyConfig : null,
  );

  /* v8 ignore start — GitHub Releases download + stream require network access */
  // Stream download zip
  const fetchOptions: RequestInit = {
    redirect: 'follow',
    headers: { Accept: 'application/octet-stream' },
    ...proxyFetchOpts,
  };

  let response: Response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (err) {
    // Provide a more helpful error for proxy-related failures
    restoreEnv?.();
    const message = (err as Error).message || String(err);
    if (
      proxyConfig &&
      (message.includes('proxy') ||
        message.includes('ECONNREFUSED') ||
        message.includes('ENOTFOUND') ||
        message.includes('tunnel') ||
        message.includes('407'))
    ) {
      throw new DownloadError(
        `Failed to connect through proxy (${proxyConfig.url}): ${message}\n` +
          `Verify proxy configuration and connectivity.` +
          (proxyConfig.username ? `\nAuthenticating as: ${proxyConfig.username}` : ''),
        0,
      );
    }
    throw new DownloadError(
      `Failed to download model: ${message}\n` +
        `URL: ${url}\n` +
        `If the model is not available, export it locally:\n` +
        `  pip install "timesfm[torch]" onnx onnxruntime torch\n` +
        `  python scripts/export-onnx.py --output ${dest}`,
      0,
    );
  }

  if (!response.ok) {
    // Detect proxy authentication failures (HTTP 407) and throw the specific error type
    restoreEnv?.();
    if (response.status === 407) {
      const proxyHint = proxyConfig
        ? `\nProxy: ${proxyConfig.url}` +
          (proxyConfig.username ? ` (user: ${proxyConfig.username})` : '')
        : '';
      throw new ProxyAuthError(
        `Proxy authentication required (HTTP 407).${proxyHint}\n` +
          `Set proxy credentials via environment variables:\n` +
          `  TIMESFM_PROXY_URL=http://proxy:8080\n` +
          `  TIMESFM_PROXY_USERNAME=user\n` +
          `  TIMESFM_PROXY_PASSWORD=pass\n` +
          `Or pass them as DownloadOptions.proxy.`,
        response.status,
      );
    }
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
  const totalMB = total > 0 ? total / 1024 / 1024 : profile.expectedZipSize / 1024 / 1024;
  const totalBytes = total > 0 ? total : profile.expectedZipSize;

  const fileStream = createWriteStream(tmpZip);
  const reader = response.body?.getReader();
  if (!reader) throw new DownloadError('No response body — cannot download model', 0);

  let received = 0;
  const startTime = Date.now();
  let lastLogAt = 0;
  let lastProgressAt = 0; // throttle onProgress to at most once per 200ms

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        received += value.length;

        // Write to stream with backpressure handling.
        // Register a single error handler on the stream (not per-chunk)
        // to avoid listener accumulation that would cause memory leaks
        // and MaxListenersExceededWarning over large downloads.
        const writeOk = fileStream.write(value);
        if (!writeOk) {
          await new Promise<void>((resolve, reject) => {
            fileStream.once('drain', () => {
              fileStream.removeAllListeners('error');
              resolve();
            });
            fileStream.once('error', (err) => {
              fileStream.removeAllListeners('drain');
              reject(err);
            });
          });
        }

        const receivedMB = received / 1024 / 1024;
        const elapsed = (Date.now() - startTime) / 1000;
        const speed = elapsed > 0 ? receivedMB / elapsed : 0;

        if (options.onProgress) {
          // Throttle: call at most once per 200ms.  Each chunk is ~64KB and
          // an 885 MB download produces ~14,000 chunks — calling onProgress
          // on every chunk would flood the event loop with 10,000+ callbacks.
          const now = Date.now();
          if (now - lastProgressAt >= 200) {
            lastProgressAt = now;
            options.onProgress(receivedMB, totalMB, speed);
          }
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

    // Final progress callback — always fires at 100% regardless of throttle
    if (options.onProgress) {
      const finalMB = received / 1024 / 1024;
      const finalElapsed = (Date.now() - startTime) / 1000;
      options.onProgress(finalMB, totalMB, finalElapsed > 0 ? finalMB / finalElapsed : 0);
    }

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

    /* v8 ignore stop */

    // Read descriptor for SHA verification
    let expectedSha256: string | null = null;
    const descriptorPath = path.join(cacheDir, DESCRIPTOR_FILENAME);
    try {
      const desc = JSON.parse(fs.readFileSync(descriptorPath, 'utf-8'));
      expectedSha256 = desc?.onnx?.sha256 ?? null;
    } catch {
      // Descriptor missing — skip checksum verification
    }

    // Verify SHA-256 of extracted ONNX (async streaming for large files)
    if (expectedSha256) {
      const actualSha256 = await sha256File(dest);
      if (actualSha256 !== expectedSha256) {
        cleanupPartial(cacheDir, profile);
        restoreEnv?.();
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
    restoreEnv?.();
    return dest;
  } catch (err) {
    restoreEnv?.();
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

/**
 * Compute SHA-256 of a file asynchronously using streaming I/O.
 *
 * Uses Node.js stream pipeline to avoid blocking the event loop
 * during hash computation of large files (e.g., 885 MB ONNX model).
 *
 * Falls back to synchronous hashing for small files (< 100 MB)
 * where the event-loop overhead of async I/O isn't justified.
 */
async function sha256File(filePath: string): Promise<string> {
  const { statSync, createReadStream } = await import('node:fs');
  const { pipeline } = await import('node:stream/promises');

  const fileSize = statSync(filePath).size;
  // For files under 100 MB, sync is faster (avoids async overhead)
  if (fileSize < 100 * 1024 * 1024) {
    return sha256FileSync(filePath);
  }

  const hasher = createHash('sha256');
  const readStream = createReadStream(filePath, { highWaterMark: 64 * 1024 });

  try {
    await pipeline(readStream, hasher);
  } catch {
    // Fall back to sync on stream failure (e.g., permission issues)
    return sha256FileSync(filePath);
  }

  return hasher.digest('hex');
}

/** Synchronous SHA-256 fallback for small files or stream failure. */
function sha256FileSync(filePath: string): string {
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
function cleanupPartial(cacheDir: string, profile: PrecisionProfile): void {
  for (const f of [profile.onnxFilename, DESCRIPTOR_FILENAME]) {
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
function isModelCachedAtPath(p: string, minSize?: number): boolean {
  if (!existsSync(p)) return false;
  try {
    return fs.statSync(p).size >= (minSize ?? 800 * 1024 * 1024);
  } catch {
    return false;
  }
}

/**
 * Check if the default model is already cached.
 */
export function isModelCached(): boolean {
  const cached = getCachedModelPath();
  return cached !== null;
}
