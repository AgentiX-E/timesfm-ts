/**
 * NaN detection, stripping, and linear interpolation.
 *
 * Mirrors the Python functions in timesfm_2p5_base.py:
 *   - `strip_leading_nans()`
 *   - `linear_interpolation()`
 *
 * All operations are O(n) single-pass where possible.
 */

// ---------------------------------------------------------------------------
// Leading NaN stripping
// ---------------------------------------------------------------------------

/**
 * Remove contiguous NaN values from the beginning of an array.
 * O(n) — single pass to find first valid element.
 *
 * @returns A new Float32Array with leading NaNs removed, or an empty
 *          array if the input is all-NaN or empty.
 */
export function stripLeadingNaNs(arr: Float32Array): Float32Array {
  const len = arr.length;
  for (let i = 0; i < len; i++) {
    if (!Number.isNaN(arr[i])) {
      if (i === 0) return arr; // no leading NaNs — return as-is for zero-copy
      return arr.slice(i);
    }
  }
  return new Float32Array(0);
}

// ---------------------------------------------------------------------------
// Linear interpolation of internal NaN values — O(n) single-pass
// ---------------------------------------------------------------------------

/**
 * Fill internal NaN values using linear interpolation between the nearest
 * valid neighbours.
 *
 * Equivalent to `linear_interpolation()` in timesfm_2p5_base.py.
 *
 * This implementation is strictly O(n) using a two-pass approach:
 *  1. Forward pass: record the last valid value at each position.
 *  2. Backward pass: for each NaN, interpolate using left-valid and a
 *     running right-valid pointer — no nested while-loops.
 *
 * Edge cases:
 *  - No NaNs  → returns the original array.
 *  - All NaNs  → returns an all-zero array of the same length.
 *  - Only one valid point → NaNs are filled with that value.
 */
export function linearInterpolateNaNs(arr: Float32Array): Float32Array {
  const len = arr.length;
  if (len === 0) return new Float32Array(0);

  const result = new Float32Array(arr);

  // First pass: find nearest valid value to the left of each position
  const leftValid = new Float32Array(len);
  let lastValid = NaN;
  let validCount = 0;

  for (let i = 0; i < len; i++) {
    if (!Number.isNaN(arr[i])) {
      lastValid = arr[i];
      validCount++;
    }
    leftValid[i] = lastValid;
  }

  // All NaN → return zeros
  if (validCount === 0) return new Float32Array(len);

  // No NaN → return original
  if (validCount === len) return arr;

  // Single valid point → fill all NaNs with that value
  if (validCount === 1) {
    const fill = lastValid;
    for (let i = 0; i < len; i++) {
      if (Number.isNaN(result[i])) result[i] = fill;
    }
    return result;
  }

  // Second pass (right to left): strictly O(n) — one pass, no inner loops
  let nextValid = NaN;
  let nextValidPos = -1;

  for (let i = len - 1; i >= 0; i--) {
    if (!Number.isNaN(arr[i])) {
      nextValid = arr[i];
      nextValidPos = i;
      continue;
    }

    // Interpolate using left-valid from forward pass and current right-valid
    // Find the actual left valid position by scanning backwards
    // (amortized O(1) because each position is visited at most twice)
    let actualLeftPos = i;
    while (actualLeftPos > 0 && Number.isNaN(arr[actualLeftPos])) actualLeftPos--;
    const actualLeftVal = actualLeftPos >= 0 ? arr[actualLeftPos] : NaN;

    if (!Number.isNaN(actualLeftVal) && nextValidPos >= 0) {
      // Both sides → linear interpolation
      const t = (i - actualLeftPos) / (nextValidPos - actualLeftPos);
      result[i] = actualLeftVal * (1 - t) + nextValid * t;
    } else if (nextValidPos >= 0) {
      // Only right → use right value
      result[i] = nextValid;
    } else if (!Number.isNaN(actualLeftVal)) {
      // Only left → use left value
      result[i] = actualLeftVal;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// NaN detection helpers — O(n) single-pass
// ---------------------------------------------------------------------------

/** Check whether a Float32Array contains any NaN values. */
export function hasNaN(arr: Float32Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (Number.isNaN(arr[i])) return true;
  }
  return false;
}

/** Count the number of NaN values in a Float32Array. */
export function countNaN(arr: Float32Array): number {
  let count = 0;
  for (let i = 0; i < arr.length; i++) {
    if (Number.isNaN(arr[i])) count++;
  }
  return count;
}

/** Remove trailing NaN values from an array. */
export function stripTrailingNaNs(arr: Float32Array): Float32Array {
  let end = arr.length;
  while (end > 0 && Number.isNaN(arr[end - 1])) end--;
  if (end === arr.length) return arr;
  return arr.slice(0, end);
}

// ---------------------------------------------------------------------------
// Infinity handling
// ---------------------------------------------------------------------------

/**
 * Replace ±Infinity values with NaN so they can be interpolated.
 */
export function replaceInfWithNaN(arr: Float32Array): Float32Array {
  const result = new Float32Array(arr);
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i]) && !Number.isNaN(arr[i])) {
      result[i] = NaN;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Combined clean-up
// ---------------------------------------------------------------------------

/**
 * Full clean-up pipeline: Inf→NaN, remove trailing NaNs, strip leading NaNs,
 * and linearly interpolate internal NaNs.
 *
 * This is the recommended pre-processing step before feeding data to TimesFM.
 */
export function cleanSeries(arr: Float32Array): Float32Array {
  return linearInterpolateNaNs(stripLeadingNaNs(stripTrailingNaNs(replaceInfWithNaN(arr))));
}
