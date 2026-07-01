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
    if (!Number.isNaN(arr[i]!)) {
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
 * This implementation is **strictly O(n)** with two single-pass traversals:
 *  1. Forward pass:  record the last valid **value** and its **position**
 *     at every index into auxiliary arrays.
 *  2. Backward pass: for each NaN, interpolate using the pre-recorded
 *     left-valid (value & position) and a running right-valid pointer.
 *     No inner loops — each array element is visited exactly twice in total.
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

  // Forward pass: record both last valid value and its position
  const leftValid = new Float32Array(len);
  const leftPos = new Int32Array(len);
  let lastValid = NaN;
  let lastPos = -1;
  let validCount = 0;

  for (let i = 0; i < len; i++) {
    if (!Number.isNaN(arr[i]!)) {
      lastValid = arr[i]!;
      lastPos = i;
      validCount++;
    }
    leftValid[i] = lastValid;
    leftPos[i] = lastPos;
  }

  // All NaN → return zeros
  if (validCount === 0) return new Float32Array(len);

  // No NaN → return original
  if (validCount === len) return arr;

  // Single valid point → fill all NaNs with that value
  if (validCount === 1) {
    const fill = lastValid;
    for (let i = 0; i < len; i++) {
      if (Number.isNaN(result[i]!)) result[i] = fill;
    }
    return result;
  }

  // Backward pass: strictly O(n) — no inner while-loop, every element visited
  // at most once.  leftValid[i] and leftPos[i] were pre-computed above.
  let rightValid = NaN;
  let rightPos = -1;

  for (let i = len - 1; i >= 0; i--) {
    if (!Number.isNaN(arr[i]!)) {
      rightValid = arr[i]!;
      rightPos = i;
      continue;
    }

    const lVal = leftValid[i]!;
    const lPos = leftPos[i]!;

    if (!Number.isNaN(lVal) && rightPos >= 0) {
      // Both sides valid → linear interpolation
      const t = (i - lPos) / (rightPos - lPos);
      result[i] = lVal * (1 - t) + rightValid * t;
    } else if (rightPos >= 0) {
      // Only right side → use right value
      result[i] = rightValid;
    } else if (!Number.isNaN(lVal)) {
      // Only left side → use left value
      result[i] = lVal;
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
