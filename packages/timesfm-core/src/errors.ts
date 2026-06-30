/**
 * Typed error hierarchy for agentix-timesfm-ts.
 *
 * All project errors extend `TimesFMError` so callers can catch them
 * with a single `instanceof` check.  Each subclass carries structured
 * context (HTTP status, field name, etc.) for programmatic handling.
 *
 * Backward-compatible: every error class extends the built-in `Error`,
 * so existing `instanceof Error` / `rejects.toThrow()` assertions are
 * unaffected.
 */

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

/** Base class for all TimesFM errors. */
export class TimesFMError extends Error {
  constructor(message?: string) {
    super(message);
    this.name = 'TimesFMError';
  }
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

/** The model has not been compiled yet. */
export class ModelNotCompiledError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'ModelNotCompiledError';
  }
}

/** The model path is required but was not provided. */
export class ModelNotFoundError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'ModelNotFoundError';
  }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Forecast configuration validation failed. */
export class ConfigValidationError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'ConfigValidationError';
  }
}

/** The requested horizon exceeds the compiled maxHorizon. */
export class HorizonExceededError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'HorizonExceededError';
  }
}

// ---------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------

/** Generic download failure (HTTP error, network, etc.). */
export class DownloadError extends TimesFMError {
  constructor(
    message: string,
    public readonly httpStatus: number,
  ) {
    super(message);
    this.name = 'DownloadError';
  }
}

/** Proxy authentication failed (HTTP 407). */
export class ProxyAuthError extends DownloadError {
  constructor(message: string, httpStatus: number) {
    super(message, httpStatus);
    this.name = 'ProxyAuthError';
  }
}

/** Downloaded file checksum does not match the expected value. */
export class ChecksumMismatchError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'ChecksumMismatchError';
  }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/** ONNX Runtime inference failure. */
export class InferenceError extends TimesFMError {
  constructor(
    message: string,
    public readonly cause?: Error,
  ) {
    super(message);
    this.name = 'InferenceError';
  }
}

// ---------------------------------------------------------------------------
// Hierarchical reconciliation
// ---------------------------------------------------------------------------

/** Hierarchy definition validation failure (orphan nodes, cycles, etc.). */
export class HierarchyValidationError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'HierarchyValidationError';
  }
}

/** Covariate / regression engine error. */
export class CovariateError extends TimesFMError {
  constructor(message?: string) {
    super(message);
    this.name = 'CovariateError';
  }
}
