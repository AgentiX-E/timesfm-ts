/**
 * Concurrency Stress Test — verifies that ONNX Runtime InferenceSession.run()
 * is safe to call concurrently from multiple Promise chains.
 *
 * P2-1: ONNX Runtime Concurrent Session Safety
 *
 * The TimesFMModel.forecast() path runs main + flip-invariance decodes
 * concurrently via Promise.all(), and the ONNX engine processes batch
 * elements concurrently.  This test validates that these concurrent
 * session.run() calls are safe and produce identical results regardless
 * of concurrency level.
 *
 * This test REQUIRES the 885 MB TimesFM ONNX model (set TIMESFM_TEST_MODEL
 * or TIMESFM_MODEL_PATH).  It runs in the integration test suite only.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TimesFMModel, createForecastConfig } from '@agentix-e/timesfm-core';
import * as fs from 'fs';

// Use the same model resolution as globalSetup
const MODEL_PATH =
  process.env.TIMESFM_TEST_MODEL || process.env.TIMESFM_MODEL_PATH || 'models/timesfm-2.5.onnx';

const hasModel = fs.existsSync(MODEL_PATH);

describe('ONNX Runtime Concurrency Safety', () => {
  let model: TimesFMModel | null = null;

  beforeAll(async () => {
    if (!hasModel) {
      console.warn(
        `[concurrency-test] Model not found at ${MODEL_PATH} — skipping integration tests.\n` +
          `  Set TIMESFM_TEST_MODEL to your ONNX model path to enable these tests.`,
      );
      return;
    }
    model = await TimesFMModel.fromPretrained({ modelPath: MODEL_PATH });
    model.compile(createForecastConfig({ maxContext: 512, maxHorizon: 128 }));
  }, 300000);

  afterAll(async () => {
    if (model) await model.dispose();
  });

  /**
   * Test: Concurrent forecast() calls produce deterministic results.
   *
   * We run the same forecast 5 times concurrently and verify that all
   * results are identical (deterministic inference path).
   */
  it(
    'concurrent forecast() calls produce identical results',
    { skip: !hasModel },
    async () => {
      const horizon = 12;
      const input = generateTestSeries(200);

      // Run 5 concurrent forecasts with the same inputs
      const results = await Promise.all(
        Array.from({ length: 5 }, () => model!.forecast(horizon, [input])),
      );

      // All results should be identical
      const first = results[0];
      for (let i = 1; i < results.length; i++) {
        expect(results[i].pointForecast.length).toBe(first.pointForecast.length);
        for (let b = 0; b < first.pointForecast.length; b++) {
          expect(results[i].pointForecast[b].length).toBe(first.pointForecast[b].length);
          for (let h = 0; h < first.pointForecast[b].length; h++) {
            expect(results[i].pointForecast[b][h]).toBeCloseTo(first.pointForecast[b][h], 4);
          }
        }
      }
    },
    60000,
  );

  /**
   * Test: Concurrent forecasts with different inputs do not interfere.
   *
   * We run forecasts with different input series concurrently and verify
   * that each result matches the equivalent sequential run.
   */
  it(
    'concurrent forecasts with different inputs do not interfere',
    { skip: !hasModel },
    async () => {
      const horizon = 12;
      const input1 = generateTestSeries(150, 1);
      const input2 = generateTestSeries(150, 2);
      const input3 = generateTestSeries(150, 3);

      // Run concurrently
      const [r1, r2, r3] = await Promise.all([
        model!.forecast(horizon, [input1]),
        model!.forecast(horizon, [input2]),
        model!.forecast(horizon, [input3]),
      ]);

      // Run sequentially for comparison
      const [s1, s2, s3] = [
        await model!.forecast(horizon, [input1]),
        await model!.forecast(horizon, [input2]),
        await model!.forecast(horizon, [input3]),
      ];

      // Concurrent and sequential results should match
      for (let h = 0; h < horizon; h++) {
        expect(r1.pointForecast[0][h]).toBeCloseTo(s1.pointForecast[0][h], 4);
        expect(r2.pointForecast[0][h]).toBeCloseTo(s2.pointForecast[0][h], 4);
        expect(r3.pointForecast[0][h]).toBeCloseTo(s3.pointForecast[0][h], 4);
      }
    },
    60000,
  );

  /**
   * Test: Flip invariance concurrent path produces valid results.
   *
   * When forceFlipInvariance=true (default), the model runs both the main
   * and negated-input decodes concurrently via Promise.all. This test
   * verifies that the flip-invariant forecast has the expected antisymmetry:
   * forecast(-x) ≈ -forecast(x) for the median.
   */
  it(
    'flip invariance produces antisymmetric results',
    { skip: !hasModel },
    async () => {
      const horizon = 12;
      const input = generateTestSeries(200, 42);
      const negInput = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) negInput[i] = -input[i];

      const modelWithFlip = await TimesFMModel.fromPretrained({ modelPath: MODEL_PATH });
      modelWithFlip.compile(
        createForecastConfig({
          maxContext: 512,
          maxHorizon: 128,
          forceFlipInvariance: true,
          normalizeInputs: false,
        }),
      );

      const [posResult, negResult] = await Promise.all([
        modelWithFlip.forecast(horizon, [input]),
        modelWithFlip.forecast(horizon, [negInput]),
      ]);

      // For flip-invariant forecasts: f(-x) ≈ -f(x)
      for (let h = 0; h < horizon; h++) {
        expect(negResult.pointForecast[0][h]).toBeCloseTo(-posResult.pointForecast[0][h], 0);
      }

      await modelWithFlip.dispose();
    },
    120000,
  );

  /**
   * Test: High-concurrency batch (10 concurrent forecasts) does not crash.
   *
   * This is a stress test: while ONNX Runtime C++ backend may not be
   * truly thread-safe, the Node.js event loop serializes JavaScript
   * execution. We verify that 10 concurrent forecast() calls complete
   * without errors, timeouts, or crashes.
   */
  it(
    'high-concurrency batch (10) completes without errors',
    { skip: !hasModel },
    async () => {
      const horizon = 12;
      const inputs = Array.from({ length: 10 }, (_, i) => generateTestSeries(100 + i * 10, i));

      // 10 concurrent forecasts — this MUST not crash or hang
      const results = await Promise.all(inputs.map((input) => model!.forecast(horizon, [input])));

      expect(results).toHaveLength(10);
      for (const r of results) {
        expect(r.pointForecast).toBeDefined();
        expect(r.pointForecast.length).toBe(1);
        expect(r.pointForecast[0].length).toBe(horizon);
      }
    },
    120000,
  );
});

// ── Helpers ──────────────────────────────────────────────────────────────

/** Deterministic test series generator (Mulberry32 PRNG, seed-based). */
function generateTestSeries(len: number, seed = 0): Float32Array {
  let s = 42 + seed;
  const rand = () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const arr = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    arr[i] = 100 + i * 0.5 + 20 * Math.sin((2 * Math.PI * i) / 7) + (rand() - 0.5) * 10;
  }
  return arr;
}
