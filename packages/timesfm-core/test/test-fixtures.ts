/**
 * Realistic time-series test fixtures for the TimesFM test suite.
 *
 * Each fixture represents a common real-world time-series pattern:
 *   - Trend + seasonality + noise (typical business metric)
 *   - Pure seasonal (daily/weekly patterns)
 *   - Random walk (stock prices)
 *   - Spike/outlier (anomaly detection)
 *   - Constant / near-constant (degenerate cases)
 *   - Long series (stress test)
 */

// ---------------------------------------------------------------------------
// Fixture generators (deterministic — seeded pseudo-random)
// ---------------------------------------------------------------------------

/** Simple deterministic pseudo-random number generator (Mulberry32). */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEED = 42;
const rng = mulberry32(SEED);

// ---------------------------------------------------------------------------
// Fixture 1: Business metric with trend, weekly seasonality, and noise
// ---------------------------------------------------------------------------

/**
 * Simulates a realistic business metric (e.g., daily website visitors):
 *   - Linear trend: +0.5 per step
 *   - Weekly seasonality: amplitude 20, period 7
 *   - Gaussian-like noise: amplitude 5
 *
 * @param length  Number of time steps
 * @returns A Float32Array representing the time series
 */
export function businessMetric(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const trend = 100 + i * 0.5;
    const seasonal = 20 * Math.sin((2 * Math.PI * i) / 7);
    const noise = (rng() - 0.5) * 10;
    arr[i] = trend + seasonal + noise;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 2: Pure seasonal — hourly temperature (24h cycle)
// ---------------------------------------------------------------------------

/**
 * Simulates hourly temperature over multiple days:
 *   - Daily cycle: amplitude 8°C, period 24
 *   - Baseline: 20°C
 *   - Light noise: ±1°C
 */
export function hourlyTemp(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const base = 20;
    const daily = 8 * Math.sin((2 * Math.PI * (i + 6)) / 24); // peak at ~14:00
    const noise = (rng() - 0.5) * 2;
    arr[i] = base + daily + noise;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 3: Random walk (stock price simulation)
// ---------------------------------------------------------------------------

/**
 * Simulates a log-normal random walk (stock price):
 *   - Starting price: 100
 *   - Daily volatility: 1.5%
 *   - Slight upward drift: 0.02% per step
 */
export function stockPrice(length: number): Float32Array {
  const arr = new Float32Array(length);
  let price = 100;
  for (let i = 0; i < length; i++) {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.max(rng(), 1e-10);
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const dailyReturn = 0.0002 + 0.015 * z;
    price *= 1 + dailyReturn;
    arr[i] = price;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 4: Series with spikes (anomaly detection scenario)
// ---------------------------------------------------------------------------

/**
 * Steady signal with occasional extreme spikes:
 *   - Baseline: 10
 *   - Normal noise: ±0.5
 *   - Spike at positions 30, 75, 120: values jump to 50, 0, 100
 */
export function withSpikes(length: number): Float32Array {
  const arr = new Float32Array(length);
  const spikePositions = new Set([30, 75, 120]);
  const spikeValues: Record<number, number> = { 30: 50, 75: 0, 120: 100 };
  for (let i = 0; i < length; i++) {
    if (spikePositions.has(i)) {
      arr[i] = spikeValues[i];
    } else {
      arr[i] = 10 + (rng() - 0.5);
    }
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 5: Multiplicative seasonal (e-commerce)
// ---------------------------------------------------------------------------

/**
 * E-commerce-like series with multiplicative seasonality:
 *   - Growing trend
 *   - Weekly cycle with weekend boost
 *   - Yearly cycle
 */
export function eCommerce(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const trend = 1000 * (1 + i * 0.001);
    const weekly = 1 + 0.3 * Math.sin((2 * Math.PI * i) / 7); // weekend boost
    const yearly = 1 + 0.2 * Math.sin((2 * Math.PI * i) / 365);
    const noise = 1 + (rng() - 0.5) * 0.1;
    arr[i] = trend * weekly * yearly * noise;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 6: Constant / near-constant
// ---------------------------------------------------------------------------

/** Constant value — edge case for normalization. */
export function constantSeries(length: number, value: number = 42): Float32Array {
  return new Float32Array(length).fill(value);
}

/** Near-constant with tiny noise. */
export function nearConstant(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    arr[i] = 100 + (rng() - 0.5) * 1e-6;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 7: Long series (stress test)
// ---------------------------------------------------------------------------

/**
 * Long series approaching the model's context limit (16384).
 * Combined trend + multi-seasonal + noise.
 */
export function longSeries(length: number = 10000): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const trend = 50 + i * 0.002;
    const weekly = 10 * Math.sin((2 * Math.PI * i) / 7);
    const monthly = 5 * Math.sin((2 * Math.PI * i) / 30);
    const noise = (rng() - 0.5) * 3;
    arr[i] = trend + weekly + monthly + noise;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 8: Negative values (temperature in Celsius)
// ---------------------------------------------------------------------------

/**
 * Temperature-like series with negative values:
 *   - Winter baseline around -5°C
 *   - Daily cycle ±8°C
 *   - Slow upward trend (spring arrival)
 */
export function negativeValues(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const trend = -5 + i * 0.01; // slowly warming
    const daily = 8 * Math.sin((2 * Math.PI * i) / 24);
    const noise = (rng() - 0.5) * 2;
    arr[i] = trend + daily + noise;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 9: Step change (regime shift)
// ---------------------------------------------------------------------------

/**
 * Signal with a clear level shift halfway through:
 *   - First half: mean 10, noise ±2
 *   - Second half: mean 30, noise ±2
 */
export function regimeShift(length: number): Float32Array {
  const arr = new Float32Array(length);
  const midpoint = Math.floor(length / 2);
  for (let i = 0; i < length; i++) {
    const mean = i < midpoint ? 10 : 30;
    arr[i] = mean + (rng() - 0.5) * 4;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Fixture 10: Exponential growth
// ---------------------------------------------------------------------------

/**
 * Pure exponential growth (no seasonality):
 *   - Starting value: 1
 *   - Growth rate: 1% per step
 */
export function exponentialGrowth(length: number): Float32Array {
  const arr = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    arr[i] = Math.pow(1.01, i);
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Utility: all fixtures as named entries
// ---------------------------------------------------------------------------

export interface FixtureEntry {
  name: string;
  description: string;
  generator: (length: number) => Float32Array;
}

/** All available realistic test fixtures. */
export const ALL_FIXTURES: FixtureEntry[] = [
  {
    name: 'businessMetric',
    description: 'Trend + weekly seasonality + noise',
    generator: businessMetric,
  },
  { name: 'hourlyTemp', description: 'Hourly temperature (24h cycle)', generator: hourlyTemp },
  { name: 'stockPrice', description: 'Log-normal random walk', generator: stockPrice },
  { name: 'withSpikes', description: 'Steady signal with extreme spikes', generator: withSpikes },
  { name: 'eCommerce', description: 'Multiplicative seasonal (e-commerce)', generator: eCommerce },
  {
    name: 'constantSeries',
    description: 'Constant value (edge case)',
    generator: (n) => constantSeries(n),
  },
  { name: 'nearConstant', description: 'Near-constant with tiny noise', generator: nearConstant },
  {
    name: 'longSeries',
    description: 'Long series stress test (~10k points)',
    generator: longSeries,
  },
  {
    name: 'negativeValues',
    description: 'Temperature-like with negative values',
    generator: negativeValues,
  },
  { name: 'regimeShift', description: 'Step change / regime shift', generator: regimeShift },
  {
    name: 'exponentialGrowth',
    description: 'Pure exponential growth',
    generator: exponentialGrowth,
  },
];
