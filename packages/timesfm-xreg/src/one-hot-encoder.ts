/**
 * One-Hot Encoder for categorical covariates.
 *
 * Mirrors scikit-learn's `preprocessing.OneHotEncoder` with
 * `drop='first'` and `handle_unknown='ignore'`.
 *
 * This is a pure-TypeScript implementation, avoiding the need for
 * an external Python dependency in the XReg pipeline.
 */

export type Category = number | string;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OneHotEncoderState {
  /** Sorted list of unique categories seen during fit (before drop). */
  categories: Category[];
  /** Whether to drop the first category (avoid multicollinearity). */
  drop: 'first' | null;
  /** Category → index mapping (after drop). */
  indexMap: Map<string, number>;
  /** Number of output columns (after drop). */
  numColumns: number;
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

/**
 * One-Hot Encoder for categorical features.
 *
 * Usage:
 * ```typescript
 * const encoder = new OneHotEncoder({ drop: 'first' });
 * encoder.fit(['a', 'b', 'c', 'a']);
 * const encoded = encoder.transform(['b', 'd']); // [[0,1], [0,0]]
 * ```
 */
export class OneHotEncoder {
  private state: OneHotEncoderState | null = null;

  private readonly _drop: 'first' | null;
  private readonly _handleUnknown: 'ignore' | 'error';

  constructor(
    options: {
      drop?: 'first' | null;
      handleUnknown?: 'ignore' | 'error';
    } = {},
  ) {
    this._drop = options.drop !== undefined ? options.drop : 'first';
    this._handleUnknown = options.handleUnknown ?? 'ignore';
  }

  /**
   * Fit the encoder to the given categories.
   */
  fit(values: Category[]): void {
    // Get sorted unique categories
    const unique = new Set<string>();
    for (const v of values) {
      unique.add(String(v));
    }
    const sorted = Array.from(unique).sort();

    // Build index map
    const indexMap = new Map<string, number>();
    let numColumns = sorted.length;

    if (this._drop === 'first') {
      numColumns = Math.max(0, sorted.length - 1);
    }

    let colIdx = 0;
    for (let i = 0; i < sorted.length; i++) {
      if (this._drop === 'first' && i === 0) continue; // drop first
      indexMap.set(sorted[i], colIdx);
      colIdx++;
    }

    this.state = {
      categories: sorted,
      drop: this._drop,
      indexMap,
      numColumns,
    };
  }

  /**
   * Transform values into one-hot encoded rows.
   *
   * Each row is an array of length `numColumns`, with exactly one `1`
   * (or all zeros for unknown categories when `handleUnknown='ignore'`).
   */
  transform(values: Category[]): number[][] {
    if (!this.state) {
      throw new Error('OneHotEncoder not fitted. Call fit() first.');
    }

    const { indexMap, numColumns } = this.state;
    const result: number[][] = [];

    for (const value of values) {
      const row = new Array<number>(numColumns).fill(0);
      const key = String(value);
      const idx = indexMap.get(key);

      if (idx !== undefined) {
        row[idx] = 1;
      } else if (this._handleUnknown === 'error') {
        throw new Error(`Unknown category: "${value}"`);
      }
      // else: handleUnknown='ignore' → all zeros

      result.push(row);
    }

    return result;
  }

  /**
   * Fit and transform in one call.
   */
  fitTransform(values: Category[]): number[][] {
    this.fit(values);
    return this.transform(values);
  }

  /**
   * Number of output columns after encoding.
   */
  get numColumns(): number {
    return this.state?.numColumns ?? 0;
  }
}
