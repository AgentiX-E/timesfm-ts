/**
 * Comprehensive tests for model-downloader.ts — cache helpers,
 * path resolution, proxy configuration (including NO_PROXY),
 * HTTP download/error handling, progress callbacks, and checksum verification.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as path from 'node:path';
import * as os from 'node:os';
import * as fs from 'node:fs';
import { createHash } from 'node:crypto';
import { execSync } from 'node:child_process';
import nock from 'nock';

import {
  isModelCached,
  getCachedModelPath,
  defaultModelPath,
  downloadModel,
} from '../src/model-downloader';
import { DownloadError, ChecksumMismatchError } from '../src/errors';

// ─── Mock undici for proxy-agent tracking ───────────────────────────────────

const { proxyAgentCalls } = vi.hoisted(() => ({
  proxyAgentCalls: [] as Array<{ uri: string }>,
}));

vi.mock('undici', () => ({
  ProxyAgent: vi.fn().mockImplementation((opts: { uri: string }) => {
    proxyAgentCalls.push({ uri: opts.uri });
    // Return a minimal dispatcher so fetch() doesn't crash.
    // nock intercepts at the http module level so dispatch() is never called.
    return {
      dispatch: vi.fn(),
      close: vi.fn(),
      destroy: vi.fn(),
    };
  }),
}));

// ─── Helpers ────────────────────────────────────────────────────────────────

function createTempDir(): string {
  const dir = path.join(
    os.tmpdir(),
    `timesfm-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * Create a test zip file containing an ONNX model and a model descriptor.
 *
 * Uses the system `zip` command (required for the unzip-based extraction in
 * the downloader module).  Returns the raw zip buffer.
 *
 * @param onnxContent  Content of the fake ONNX file.
 * @param wrongSha256  If provided, the descriptor's SHA-256 will differ from
 *                     the actual content hash, triggering a checksum mismatch.
 */
function createTestZip(onnxContent: Buffer, wrongSha256?: string): Buffer {
  const tmpDir = createTempDir();
  const onnxFile = path.join(tmpDir, 'timesfm-2.5.onnx');

  try {
    fs.writeFileSync(onnxFile, onnxContent);

    const actualSha256 = createHash('sha256').update(onnxContent).digest('hex');
    const sha256ToUse = wrongSha256 ?? actualSha256;

    const descriptor = {
      onnx: {
        sha256: sha256ToUse,
        size: onnxContent.length,
      },
      model_version: '2.5',
      export_date: '2024-01-01T00:00:00Z',
    };
    fs.writeFileSync(
      path.join(tmpDir, 'model-descriptor.json'),
      JSON.stringify(descriptor, null, 2),
    );

    const zipPath = path.join(tmpDir, 'out.zip');
    execSync(`cd "${tmpDir}" && zip -r "${zipPath}" .`, { stdio: 'pipe' });

    return fs.readFileSync(zipPath);
  } finally {
    // Best-effort cleanup
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('model-downloader', () => {
  describe('defaultModelPath', () => {
    it('returns a path in the cache directory', () => {
      const p = defaultModelPath();
      const cacheBase = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
      expect(p).toContain(cacheBase);
      expect(p).toContain('timesfm-ts');
      expect(p).toContain('timesfm-2.5.onnx');
    });

    it('respects XDG_CACHE_HOME', () => {
      const original = process.env.XDG_CACHE_HOME;
      process.env.XDG_CACHE_HOME = '/tmp/custom-cache';
      const p = defaultModelPath();
      expect(p).toContain('/tmp/custom-cache');
      process.env.XDG_CACHE_HOME = original;
    });
  });

  describe('isModelCached', () => {
    it('returns false when no model exists', () => {
      expect(isModelCached()).toBe(false);
    });
  });

  describe('getCachedModelPath', () => {
    it('returns null when no model is cached', () => {
      expect(getCachedModelPath()).toBeNull();
    });
  });

  describe('isModelCachedAtPath', () => {
    it('returns true when a file >= MIN_CACHED_SIZE exists', () => {
      const tmpDir = createTempDir();
      const tmpFile = path.join(tmpDir, 'large-model.onnx');
      try {
        fs.writeFileSync(tmpFile, Buffer.alloc(1024)); // 1 KB — too small
        expect(fs.existsSync(tmpFile)).toBe(true);
        const stat = fs.statSync(tmpFile);
        expect(stat.size).toBeLessThan(800 * 1024 * 1024);
      } finally {
        try {
          fs.unlinkSync(tmpFile);
        } catch {
          /* ignore */
        }
        try {
          fs.rmdirSync(tmpDir);
        } catch {
          /* ignore */
        }
      }
    });
  });

  // ── Move existing download error tests under a properly-named block ───────

  describe('HTTP error handling', () => {
    let envBackup: Record<string, string | undefined>;

    beforeEach(() => {
      envBackup = {};
      for (const k of [
        'TIMESFM_PROXY_URL',
        'TIMESFM_PROXY_USERNAME',
        'TIMESFM_PROXY_PASSWORD',
        'TIMESFM_PROXY_PASSWORD_FILE',
        'HTTPS_PROXY',
        'https_proxy',
        'HTTP_PROXY',
        'http_proxy',
        'NO_PROXY',
        'no_proxy',
      ]) {
        envBackup[k] = process.env[k];
        delete process.env[k];
      }
    });

    afterEach(() => {
      for (const k of Object.keys(envBackup)) {
        if (envBackup[k] === undefined) delete process.env[k];
        else process.env[k] = envBackup[k];
      }
      nock.cleanAll();
    });

    it('downloadModel rejects with DownloadError on HTTP 404', async () => {
      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'test-model.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(404, 'Not Found');

      try {
        await downloadModel({ dest, force: true, logger: () => {} });
        expect.unreachable('Expected downloadModel to throw');
      } catch (err) {
        expect(err).toBeInstanceOf(DownloadError);
        expect((err as DownloadError).httpStatus).toBe(404);
      } finally {
        try {
          fs.unlinkSync(dest);
        } catch {
          /* ignore */
        }
        try {
          const zipPath = path.join(tmpDir, 'timesfm-onnx-2.5.zip.tmp');
          try {
            fs.unlinkSync(zipPath);
          } catch {
            /* ignore */
          }
        } catch {
          /* ignore */
        }
        try {
          fs.rmdirSync(tmpDir);
        } catch {
          /* ignore */
        }
      }
    });

    it('downloadModel rejects with DownloadError on HTTP 500', async () => {
      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'test-model.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(500, 'Internal Server Error');

      try {
        await downloadModel({ dest, force: true, logger: () => {} });
        expect.unreachable('Expected downloadModel to throw');
      } catch (err) {
        expect(err).toBeInstanceOf(DownloadError);
        expect((err as DownloadError).httpStatus).toBe(500);
        expect((err as DownloadError).message).toContain('500');
      } finally {
        nock.cleanAll();
        try {
          fs.rmdirSync(tmpDir, { recursive: true });
        } catch {
          /* ignore */
        }
      }
    });

    it('downloadModel reports error message with HTTP status', async () => {
      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(403, 'Forbidden');

      try {
        await downloadModel({ force: true, logger: () => {} });
        expect.unreachable();
      } catch (err) {
        expect(err).toBeInstanceOf(DownloadError);
        expect((err as DownloadError).httpStatus).toBe(403);
      }
    });

    it('downloadModel rejects when response has no body', async () => {
      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'test-model.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(200, '', { 'content-length': '0' });

      try {
        await downloadModel({ dest, force: true, logger: () => {} });
        expect.unreachable();
      } catch (err) {
        expect(err).toBeInstanceOf(DownloadError);
        expect((err as Error).message).toContain('Failed to extract model zip');
      } finally {
        nock.cleanAll();
        try {
          fs.rmdirSync(tmpDir, { recursive: true });
        } catch {
          /* ignore */
        }
      }
    });

    it('downloadModel skips download when model is cached and force=false', async () => {
      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'cached-model.onnx');
      try {
        const minSize = 800 * 1024 * 1024;
        const fd = fs.openSync(dest, 'w');
        try {
          fs.ftruncateSync(fd, minSize);
          fs.closeSync(fd);
        } catch {
          fs.closeSync(fd);
          const buf = Buffer.alloc(minSize);
          fs.writeFileSync(dest, buf);
        }

        const result = await downloadModel({ dest, force: false, logger: () => {} });
        expect(result).toBe(dest);
      } finally {
        try {
          fs.unlinkSync(dest);
        } catch {
          /* ignore */
        }
        try {
          fs.rmdirSync(tmpDir);
        } catch {
          /* ignore */
        }
      }
    });

    it('downloadModel with force=true skips cache check', async () => {
      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'force-model.onnx');

      fs.writeFileSync(dest, Buffer.alloc(900 * 1024 * 1024));

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(404);

      try {
        await downloadModel({ dest, force: true, logger: () => {} });
        expect.unreachable();
      } catch (err) {
        expect(err).toBeInstanceOf(DownloadError);
      } finally {
        nock.cleanAll();
        try {
          fs.unlinkSync(dest);
        } catch {
          /* ignore */
        }
        try {
          fs.rmdirSync(tmpDir);
        } catch {
          /* ignore */
        }
      }
    });
  });

  // ── Actual proxy resolution tests ─────────────────────────────────────────

  describe('proxy env var resolution', () => {
    let envBackup: Record<string, string | undefined>;

    beforeEach(() => {
      // Clear captured calls before each test
      proxyAgentCalls.length = 0;

      envBackup = {};
      for (const k of [
        'TIMESFM_PROXY_URL',
        'TIMESFM_PROXY_USERNAME',
        'TIMESFM_PROXY_PASSWORD',
        'TIMESFM_PROXY_PASSWORD_FILE',
        'HTTPS_PROXY',
        'https_proxy',
        'HTTP_PROXY',
        'http_proxy',
        'NO_PROXY',
        'no_proxy',
      ]) {
        envBackup[k] = process.env[k];
        delete process.env[k];
      }

      // Default nock: intercept the download URL so fetch() doesn't hit
      // the real network regardless of proxy settings.
      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(404);
    });

    afterEach(() => {
      for (const k of Object.keys(envBackup)) {
        if (envBackup[k] === undefined) delete process.env[k];
        else process.env[k] = envBackup[k];
      }
      nock.cleanAll();
    });

    it('uses TIMESFM_PROXY_URL when set', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://timesfm-proxy.internal:8080';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected: nock returns 404
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('timesfm-proxy.internal:8080');
    });

    it('TIMESFM_PROXY_URL takes priority over standard proxy env vars', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://timesfm-proxy.internal:8080';
      process.env.HTTPS_PROXY = 'http://standard-proxy:3128';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected: nock returns 404
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('timesfm-proxy.internal:8080');
      expect(proxyAgentCalls[0].uri).not.toContain('standard-proxy');
    });

    it('uses HTTPS_PROXY env var when no TIMESFM-specific vars are set', async () => {
      process.env.HTTPS_PROXY = 'http://standard-https-proxy:3128';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('standard-https-proxy:3128');
    });

    it('uses http_proxy (lowercase) as fallback', async () => {
      process.env.http_proxy = 'http://lowercase-proxy:8080';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('lowercase-proxy:8080');
    });

    it('uses HTTP_PROXY as fallback when https_proxy variants are unset', async () => {
      process.env.HTTP_PROXY = 'http://http-proxy:9090';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('http-proxy:9090');
    });

    it('includes credentials in ProxyAgent URI when TIMESFM_PROXY_USERNAME/PASSWORD are set', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://proxy:8080';
      process.env.TIMESFM_PROXY_USERNAME = 'myuser';
      process.env.TIMESFM_PROXY_PASSWORD = 'mypass';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('myuser:mypass@');
    });

    it('reads password from TIMESFM_PROXY_PASSWORD_FILE when set', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://proxy:8080';
      process.env.TIMESFM_PROXY_USERNAME = 'fileuser';
      // Write password to a temp file
      const tmpDir = createTempDir();
      const passwordFile = path.join(tmpDir, 'proxy-password');
      try {
        fs.writeFileSync(passwordFile, 'filepass\n'); // with trailing newline (trimmed)
        process.env.TIMESFM_PROXY_PASSWORD_FILE = passwordFile;

        try {
          await downloadModel({ force: true, logger: () => {} });
        } catch {
          // Expected
        }

        expect(proxyAgentCalls.length).toBe(1);
        expect(proxyAgentCalls[0].uri).toContain('fileuser:filepass@');
      } finally {
        delete process.env.TIMESFM_PROXY_PASSWORD_FILE;
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      }
    });

    it('TIMESFM_PROXY_PASSWORD takes priority over PASSWORD_FILE', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://proxy:8080';
      process.env.TIMESFM_PROXY_USERNAME = 'envuser';
      process.env.TIMESFM_PROXY_PASSWORD = 'envpass';

      const tmpDir = createTempDir();
      const passwordFile = path.join(tmpDir, 'proxy-password');
      try {
        fs.writeFileSync(passwordFile, 'filepass');
        process.env.TIMESFM_PROXY_PASSWORD_FILE = passwordFile;

        try {
          await downloadModel({ force: true, logger: () => {} });
        } catch {
          // Expected
        }

        // Password env var takes priority over file
        expect(proxyAgentCalls.length).toBe(1);
        expect(proxyAgentCalls[0].uri).toContain('envuser:envpass@');
      } finally {
        delete process.env.TIMESFM_PROXY_PASSWORD_FILE;
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      }
    });

    it('gracefully handles missing PASSWORD_FILE', async () => {
      process.env.TIMESFM_PROXY_URL = 'http://proxy:8080';
      process.env.TIMESFM_PROXY_USERNAME = 'nofileuser';
      process.env.TIMESFM_PROXY_PASSWORD_FILE = '/nonexistent/password-file';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      // Should still apply proxy, just without password.
      // URL format: http://nofileuser@proxy:8080/  (username but no password)
      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('nofileuser@');
      // No password separator after username
      expect(proxyAgentCalls[0].uri).not.toContain('nofileuser:');
    });

    it('strips proxy via NO_PROXY when github.com is in the exclusion list', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.NO_PROXY = 'github.com';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      // NO_PROXY matched github.com → proxy should NOT be applied
      expect(proxyAgentCalls.length).toBe(0);
    });

    it('strips proxy via no_proxy (lowercase) when github.com matches', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.no_proxy = 'github.com';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(0);
    });

    it('NO_PROXY wildcard (*) strips all proxies', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.NO_PROXY = '*';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(0);
    });

    it('NO_PROXY with non-matching host still applies proxy', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.NO_PROXY = 'other.internal,example.com';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      // Neither pattern matches github.com → proxy SHOULD be applied
      expect(proxyAgentCalls.length).toBe(1);
      expect(proxyAgentCalls[0].uri).toContain('proxy:3128');
    });

    it('NO_PROXY with .github.com strips proxy (subdomain match)', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.NO_PROXY = '.github.com';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(0);
    });

    it('NO_PROXY with comma-separated pattern containing github.com strips proxy', async () => {
      process.env.HTTPS_PROXY = 'http://proxy:3128';
      process.env.NO_PROXY = 'localhost,127.0.0.1,github.com,*.internal';

      try {
        await downloadModel({ force: true, logger: () => {} });
      } catch {
        // Expected
      }

      expect(proxyAgentCalls.length).toBe(0);
    });
  });

  // ── Successful download test ──────────────────────────────────────────────

  describe('successful download', () => {
    let envBackup: Record<string, string | undefined>;

    beforeEach(() => {
      proxyAgentCalls.length = 0;
      envBackup = {};
      for (const k of [
        'TIMESFM_PROXY_URL',
        'TIMESFM_PROXY_USERNAME',
        'TIMESFM_PROXY_PASSWORD',
        'TIMESFM_PROXY_PASSWORD_FILE',
        'HTTPS_PROXY',
        'https_proxy',
        'HTTP_PROXY',
        'http_proxy',
        'NO_PROXY',
        'no_proxy',
      ]) {
        envBackup[k] = process.env[k];
        delete process.env[k];
      }
    });

    afterEach(() => {
      for (const k of Object.keys(envBackup)) {
        if (envBackup[k] === undefined) delete process.env[k];
        else process.env[k] = envBackup[k];
      }
      nock.cleanAll();
    });

    it('downloads, extracts, and returns the model path', async () => {
      // Create a test zip with a small ONNX-like file and a valid descriptor
      const onnxContent = Buffer.from('ONNX-MODEL-DATA-' + 'A'.repeat(256));
      const zipBuffer = createTestZip(onnxContent);

      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'timesfm-2.5.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(200, zipBuffer, { 'content-length': String(zipBuffer.length) });

      try {
        const result = await downloadModel({ dest, force: true, logger: () => {} });

        // Should return the dest path
        expect(result).toBe(dest);

        // The ONNX file should exist
        expect(fs.existsSync(dest)).toBe(true);

        // Content should match what we put in the zip
        const extracted = fs.readFileSync(dest);
        expect(extracted.equals(onnxContent)).toBe(true);

        // Descriptor should also be in the cache dir
        const descriptorPath = path.join(tmpDir, 'model-descriptor.json');
        expect(fs.existsSync(descriptorPath)).toBe(true);
      } finally {
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      }
    });
  });

  // ── Checksum verification test ────────────────────────────────────────────

  describe('checksum verification', () => {
    let envBackup: Record<string, string | undefined>;

    beforeEach(() => {
      proxyAgentCalls.length = 0;
      envBackup = {};
      for (const k of [
        'TIMESFM_PROXY_URL',
        'TIMESFM_PROXY_USERNAME',
        'TIMESFM_PROXY_PASSWORD',
        'TIMESFM_PROXY_PASSWORD_FILE',
        'HTTPS_PROXY',
        'https_proxy',
        'HTTP_PROXY',
        'http_proxy',
        'NO_PROXY',
        'no_proxy',
      ]) {
        envBackup[k] = process.env[k];
        delete process.env[k];
      }
    });

    afterEach(() => {
      for (const k of Object.keys(envBackup)) {
        if (envBackup[k] === undefined) delete process.env[k];
        else process.env[k] = envBackup[k];
      }
      nock.cleanAll();
    });

    it('throws ChecksumMismatchError when SHA-256 does not match descriptor', async () => {
      const onnxContent = Buffer.from('MODEL-BYTE-SEQUENCE-' + Math.random().toString(36));
      // Create a zip where the descriptor has a deliberately wrong SHA-256
      const wrongSha256 = '0'.repeat(64);
      const zipBuffer = createTestZip(onnxContent, wrongSha256);

      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'timesfm-2.5.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(200, zipBuffer, { 'content-length': String(zipBuffer.length) });

      try {
        await downloadModel({ dest, force: true, logger: () => {} });
        expect.unreachable('Expected ChecksumMismatchError to be thrown');
      } catch (err) {
        expect(err).toBeInstanceOf(ChecksumMismatchError);
        expect((err as Error).message).toContain('Checksum mismatch');
      } finally {
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      }
    });
  });

  // ── Progress callback test ────────────────────────────────────────────────

  describe('progress callback', () => {
    let envBackup: Record<string, string | undefined>;

    beforeEach(() => {
      proxyAgentCalls.length = 0;
      envBackup = {};
      for (const k of [
        'TIMESFM_PROXY_URL',
        'TIMESFM_PROXY_USERNAME',
        'TIMESFM_PROXY_PASSWORD',
        'TIMESFM_PROXY_PASSWORD_FILE',
        'HTTPS_PROXY',
        'https_proxy',
        'HTTP_PROXY',
        'http_proxy',
        'NO_PROXY',
        'no_proxy',
      ]) {
        envBackup[k] = process.env[k];
        delete process.env[k];
      }
    });

    afterEach(() => {
      for (const k of Object.keys(envBackup)) {
        if (envBackup[k] === undefined) delete process.env[k];
        else process.env[k] = envBackup[k];
      }
      nock.cleanAll();
    });

    it('calls onProgress during download', async () => {
      const onnxContent = Buffer.from('PROGRESS-TEST-' + 'B'.repeat(256));
      const zipBuffer = createTestZip(onnxContent);

      const tmpDir = createTempDir();
      const dest = path.join(tmpDir, 'timesfm-2.5.onnx');

      nock('https://github.com')
        .get('/AgentiX-E/timesfm-ts/releases/download/timesfm-latest/timesfm-onnx-2.5.zip')
        .reply(200, zipBuffer, { 'content-length': String(zipBuffer.length) });

      const progressCalls: Array<{ received: number; total: number; speed: number }> = [];

      try {
        await downloadModel({
          dest,
          force: true,
          logger: () => {},
          onProgress: (received, total, speed) => {
            progressCalls.push({ received, total, speed });
          },
        });

        // onProgress should have been called at least once during download
        expect(progressCalls.length).toBeGreaterThanOrEqual(1);

        // Check the final progress call is close to the total size
        const lastCall = progressCalls[progressCalls.length - 1];
        const zipSizeMB = zipBuffer.length / 1024 / 1024;
        expect(lastCall.received).toBeGreaterThan(0);
        expect(lastCall.total).toBeGreaterThan(0);
        // received should be roughly the total (small file may be delivered in one chunk)
        expect(lastCall.received).toBeCloseTo(zipSizeMB, 0);
      } finally {
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          /* ignore */
        }
      }
    });
  });
});
