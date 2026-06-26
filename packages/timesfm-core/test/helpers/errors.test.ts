/**
 * Tests for the typed error hierarchy.
 *
 * Verifies that all error classes are instantiable, have correct
 * prototype chains for instanceof checks, and carry structured data.
 */

import { describe, it, expect } from 'vitest';
import {
  TimesFMError,
  ModelNotCompiledError,
  ModelNotFoundError,
  ConfigValidationError,
  HorizonExceededError,
  DownloadError,
  ProxyAuthError,
  ChecksumMismatchError,
} from '@agentix-e/timesfm-core';

describe('Typed error hierarchy', () => {
  // ── instanceof checks ─────────────────────────────────────────────────

  it('TimesFMError is instance of Error and TimesFMError', () => {
    const e = new TimesFMError('test');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e.name).toBe('TimesFMError');
    expect(e.message).toBe('test');
  });

  it('ModelNotCompiledError chain', () => {
    const e = new ModelNotCompiledError('not ready');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(ModelNotCompiledError);
    expect(e).not.toBeInstanceOf(ModelNotFoundError);
  });

  it('ModelNotFoundError chain', () => {
    const e = new ModelNotFoundError();
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(ModelNotFoundError);
  });

  it('ConfigValidationError chain', () => {
    const e = new ConfigValidationError('bad config');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(ConfigValidationError);
  });

  it('HorizonExceededError chain', () => {
    const e = new HorizonExceededError('horizon 500 > max 256');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(HorizonExceededError);
  });

  it('DownloadError carries httpStatus', () => {
    const e = new DownloadError('Not Found', 404);
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(DownloadError);
    expect(e.httpStatus).toBe(404);
    expect(e.name).toBe('DownloadError');
  });

  it('DownloadError with httpStatus 0', () => {
    const e = new DownloadError('network error', 0);
    expect(e.httpStatus).toBe(0);
  });

  it('ProxyAuthError chain and inheritance', () => {
    const e = new ProxyAuthError('proxy auth required', 407);
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(DownloadError);
    expect(e).toBeInstanceOf(ProxyAuthError);
    expect(e.httpStatus).toBe(407);
    expect(e.name).toBe('ProxyAuthError');
  });

  it('ChecksumMismatchError chain', () => {
    const e = new ChecksumMismatchError('sha256 mismatch');
    expect(e).toBeInstanceOf(Error);
    expect(e).toBeInstanceOf(TimesFMError);
    expect(e).toBeInstanceOf(ChecksumMismatchError);
    expect(e).not.toBeInstanceOf(DownloadError);
  });

  it('catch by base TimesFMError', () => {
    const allErrors: TimesFMError[] = [
      new ModelNotCompiledError(),
      new ModelNotFoundError(),
      new ConfigValidationError(),
      new HorizonExceededError(),
      new DownloadError('', 500),
      new ProxyAuthError('', 407),
      new ChecksumMismatchError(),
    ];

    for (const e of allErrors) {
      // All should be catchable as TimesFMError
      expect(e).toBeInstanceOf(TimesFMError);
      expect(e).toBeInstanceOf(Error);
    }
  });
});
