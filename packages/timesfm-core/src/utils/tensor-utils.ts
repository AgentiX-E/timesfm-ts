/**
 * Low-level tensor utility helpers.
 *
 * These handle operations that, in the Python code, are done by
 * PyTorch/NumPy broadcasting, reshaping, and slicing.  We implement them
 * manually on top of Float32Array for zero external dependencies.
 */

// ---------------------------------------------------------------------------
// Reshaping / flattening
// ---------------------------------------------------------------------------

/**
 * Reshape a flat array into a 2-D matrix of `[rows, cols]`.
 *
 * Returns an array of `rows` Float32Arrays each of length `cols`.
 */
export function reshape2D(flat: Float32Array, rows: number, cols: number): Float32Array[] {
  const result: Float32Array[] = [];
  for (let r = 0; r < rows; r++) {
    result.push(flat.slice(r * cols, (r + 1) * cols));
  }
  return result;
}

/**
 * Reshape a flat array into a 3-D tensor of `[d0, d1, d2]`.
 */
export function reshape3D(
  flat: Float32Array,
  d0: number,
  d1: number,
  d2: number,
): Float32Array[][] {
  const result: Float32Array[][] = [];
  const stride = d1 * d2;
  for (let i = 0; i < d0; i++) {
    const slice = flat.slice(i * stride, (i + 1) * stride);
    result.push(reshape2D(slice, d1, d2));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

/**
 * Left-pad an array to length `targetLen` with zeros, returning a mask
 * where 1 = padding, 0 = original value.
 */
export function leftPad(
  arr: Float32Array,
  targetLen: number,
): { padded: Float32Array; mask: Uint8Array } {
  const arrLen = arr.length;
  if (arrLen > targetLen) {
    return {
      padded: arr.slice(arrLen - targetLen),
      mask: new Uint8Array(targetLen), // all zeros (no padding)
    };
  }
  if (arrLen === targetLen) {
    // Return the array as-is to avoid unnecessary O(n) copy
    return {
      padded: arr,
      mask: new Uint8Array(targetLen), // all zeros (no padding)
    };
  }

  const padLen = targetLen - arrLen;
  const padded = new Float32Array(targetLen);
  const mask = new Uint8Array(targetLen);

  // Fill padding
  for (let i = 0; i < padLen; i++) {
    mask[i] = 1;
  }

  // Copy original values
  padded.set(arr, padLen);

  return { padded, mask };
}

// ---------------------------------------------------------------------------
// Concatenation
// ---------------------------------------------------------------------------

/**
 * Concatenate an array of Float32Arrays into a single flat Float32Array.
 */
export function concat(arrays: Float32Array[]): Float32Array {
  const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

/**
 * Concatenate an array of Uint8Arrays into a single flat Uint8Array.
 */
export function concatUint8(arrays: Uint8Array[]): Uint8Array {
  const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

/**
 * Concatenate along a new axis (stacks arrays as rows).
 * Returns a flat Float32Array where result[i * colLen + j] = arrays[i][j].
 */
export function stack(arrays: Float32Array[]): Float32Array {
  if (arrays.length === 0) return new Float32Array(0);
  const colLen = arrays[0]!.length;
  const result = new Float32Array(arrays.length * colLen);
  for (let i = 0; i < arrays.length; i++) {
    if (arrays[i]!.length !== colLen) {
      throw new RangeError(
        `stack: all arrays must have the same length (expected ${colLen}, got ${arrays[i]!.length} at index ${i})`,
      );
    }
    result.set(arrays[i]!, i * colLen);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Slicing
// ---------------------------------------------------------------------------

/**
 * Extract a slice from each array in a batch, returning new arrays.
 */
export function sliceEach(arrays: Float32Array[], start: number, end?: number): Float32Array[] {
  return arrays.map((arr) => arr.slice(start, end));
}

/**
 * Take the last N elements from each array in a batch.
 */
export function takeLast(arrays: Float32Array[], n: number): Float32Array[] {
  return arrays.map((arr) => arr.slice(Math.max(0, arr.length - n)));
}

// ---------------------------------------------------------------------------
// Clip / clamp
// ---------------------------------------------------------------------------

/**
 * Element-wise maximum with `minVal`.
 */
export function clipMin(values: Float32Array, minVal: number): Float32Array {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = Math.max(values[i]!, minVal);
  }
  return result;
}

/**
 * Element-wise minimum with `maxVal`.
 */
export function clipMax(values: Float32Array, maxVal: number): Float32Array {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = Math.min(values[i]!, maxVal);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/**
 * Element-wise mean of two arrays.
 *
 * @throws {RangeError} if the arrays have different lengths.
 */
export function elementwiseMean(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new RangeError(`Length mismatch: a.length=${a.length}, b.length=${b.length}`);
  }
  const len = a.length;
  const result = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    result[i] = (a[i]! + b[i]!) / 2;
  }
  return result;
}

/**
 * Element-wise difference: a[i] - b[i].
 *
 * @throws {RangeError} if the arrays have different lengths.
 */
export function elementwiseDiff(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new RangeError(`Length mismatch: a.length=${a.length}, b.length=${b.length}`);
  }
  const len = a.length;
  const result = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    result[i] = a[i]! - b[i]!;
  }
  return result;
}

/**
 * Negate all elements.
 */
export function negate(arr: Float32Array): Float32Array {
  const result = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = -arr[i]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/**
 * Mean of a 1-D array.  Skips NaN and non-finite values.
 *
 * For numerically-stable, masked statistics consider using `computeStats()`
 * from `./stats` instead.
 */
export function mean(arr: Float32Array): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < arr.length; i++) {
    if (Number.isFinite(arr[i]!)) {
      sum += arr[i]!;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Population standard deviation of a 1-D array.  Skips NaN and non-finite values.
 *
 * Uses a two-pass algorithm for numerical stability.  For production
 * use with weighting or masks, prefer `computeStats()` from `./stats`.
 */
export function std(arr: Float32Array): number {
  let count = 0;
  for (let i = 0; i < arr.length; i++) if (Number.isFinite(arr[i]!)) count++;
  if (count <= 1) return 0;
  const m = mean(arr);
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    if (Number.isFinite(arr[i]!)) sumSq += (arr[i]! - m) ** 2;
  }
  return Math.sqrt(Math.max(0, sumSq / count));
}

/**
 * Check if all values in a Float32Array are non-negative.
 *
 * Skips NaN values (NaN < 0 is false, so NaN would be silently treated
 * as non-negative in a naive implementation).  A series containing NaN
 * cannot be assumed positive — it should be treated as having unknown sign.
 */
export function allNonNegative(arr: Float32Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    if (Number.isNaN(v)) return false; // NaN → unknown sign, treat as not positive
    if (v < 0) return false;
  }
  return true;
}

/**
 * Check if any value in a Float32Array is NaN or Infinity.
 */
export function hasInvalid(arr: Float32Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return true;
  }
  return false;
}
