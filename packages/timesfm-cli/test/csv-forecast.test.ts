import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

// ---------------------------------------------------------------------------
// Hoisted mocks — needed so vi.mock sees callable spies
// ---------------------------------------------------------------------------

const { mockFromPretrained, mockForecast, mockCompile, mockDispose, mockCreateForecastConfig } =
  vi.hoisted(() => ({
    mockFromPretrained: vi.fn(),
    mockForecast: vi.fn(),
    mockCompile: vi.fn(),
    mockDispose: vi.fn(),
    mockCreateForecastConfig: vi.fn(),
  }));

vi.mock('@agentix/timesfm-core', () => ({
  TimesFMModel: {
    fromPretrained: mockFromPretrained,
  },
  createForecastConfig: mockCreateForecastConfig,
}));

import {
  removeTrailingNaN,
  parseCSVData,
  outputCSV,
  outputJSON,
  csvForecast,
} from '../src/csv-forecast';
import type { CSVForecastOptions } from '../src/csv-forecast';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'timesfm-test-'));
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(process.stdout, 'write').mockImplementation(() => true);
  vi.clearAllMocks();

  // Default mock return values
  mockCreateForecastConfig.mockReturnValue({
    maxContext: 512,
    maxHorizon: 12,
  });
  mockFromPretrained.mockResolvedValue({
    compile: mockCompile,
    forecast: mockForecast,
    dispose: mockDispose,
  });
  mockCompile.mockReturnValue(undefined);
  mockDispose.mockResolvedValue(undefined);
  mockForecast.mockResolvedValue(makeForecastResult());
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

function writeTempCsv(content: string): string {
  const fp = path.join(tmpDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.csv`);
  fs.writeFileSync(fp, content);
  return fp;
}

function makeOptions(overrides: Partial<CSVForecastOptions> = {}): CSVForecastOptions {
  return {
    inputPath: '/fake/input.csv',
    horizon: 3,
    modelPath: '/fake/model',
    dateCol: 'date',
    outputFormat: 'csv',
    maxContext: 512,
    normalizeInputs: false,
    forceFlipInvariance: false,
    inferIsPositive: false,
    fixQuantileCrossing: false,
    useContinuousQuantileHead: false,
    ...overrides,
  };
}

function makeForecastResult() {
  return {
    pointForecast: [new Float32Array([10.1, 20.2, 30.3])] as Float32Array[],
    quantileForecast: [
      Array.from({ length: 10 }, (_, qi) => new Float32Array([qi + 1, qi + 2, qi + 3])),
    ] as Float32Array[][],
  };
}

// ---------------------------------------------------------------------------
// removeTrailingNaN
// ---------------------------------------------------------------------------

describe('removeTrailingNaN', () => {
  it('returns original array when no trailing NaN', () => {
    const arr = new Float32Array([1, 2, 3]);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(3);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('removes single trailing NaN', () => {
    const arr = new Float32Array([1, 2, NaN]);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(2);
    expect(result[0]).toBe(1);
    expect(result[1]).toBe(2);
  });

  it('removes multiple trailing NaNs', () => {
    const arr = new Float32Array([1, NaN, NaN]);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(1);
    expect(result[0]).toBe(1);
  });

  it('returns empty array for all-NaN input', () => {
    const arr = new Float32Array([NaN, NaN, NaN]);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(0);
  });

  it('preserves internal NaN values', () => {
    const arr = new Float32Array([1, NaN, 3]);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(3);
    expect(result[0]).toBe(1);
    expect(Number.isNaN(result[1])).toBe(true);
    expect(result[2]).toBe(3);
  });

  it('handles empty array', () => {
    const arr = new Float32Array(0);
    const result = removeTrailingNaN(arr);
    expect(result.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// parseCSVData
// ---------------------------------------------------------------------------

describe('parseCSVData', () => {
  it('parses a simple CSV with date column and numeric values', () => {
    const fp = writeTempCsv(
      'date,value1,value2\n2024-01-01,10,100\n2024-01-02,20,200\n2024-01-03,30,300\n',
    );
    const result = parseCSVData(fp, 'date');

    expect(result.dates).toEqual(['2024-01-01', '2024-01-02', '2024-01-03']);
    expect(result.series.size).toBe(2);
    expect(Array.from(result.series.get('value1')!)).toEqual([10, 20, 30]);
    expect(Array.from(result.series.get('value2')!)).toEqual([100, 200, 300]);
  });

  it('uses specified valueCols to restrict parsed columns', () => {
    const fp = writeTempCsv('date,value1,value2\n2024-01-01,10,100\n2024-01-02,20,200\n');
    const result = parseCSVData(fp, 'date', ['value1']);

    expect(result.series.size).toBe(1);
    expect(result.series.has('value1')).toBe(true);
    expect(result.series.has('value2')).toBe(false);
  });

  it('throws on empty CSV', () => {
    const fp = writeTempCsv('date,value\n');
    expect(() => parseCSVData(fp, 'date')).toThrow('Empty CSV file');
  });

  it('handles numeric columns with missing values (NaN)', () => {
    const fp = writeTempCsv('date,value\n2024-01-01,10\n2024-01-02,\n2024-01-03,30\n');
    const result = parseCSVData(fp, 'date');

    expect(result.dates).toEqual(['2024-01-01', '2024-01-02', '2024-01-03']);
    const vals = Array.from(result.series.get('value')!);
    expect(vals[0]).toBe(10);
    expect(Number.isNaN(vals[1])).toBe(true);
    expect(vals[2]).toBe(30);
  });

  it('strips trailing NaN values from series', () => {
    const fp = writeTempCsv('date,value1\n2024-01-01,10\n2024-01-02,\n2024-01-03,\n');
    const result = parseCSVData(fp, 'date');

    const vals = Array.from(result.series.get('value1')!);
    expect(vals).toEqual([10]);
  });

  it('handles non-numeric text as NaN', () => {
    const fp = writeTempCsv('date,value\n2024-01-01,abc\n2024-01-02,20\n');
    const result = parseCSVData(fp, 'date');

    const vals = Array.from(result.series.get('value')!);
    expect(Number.isNaN(vals[0])).toBe(true);
    expect(vals[1]).toBe(20);
  });

  it('returns empty strings when date column is missing', () => {
    const fp = writeTempCsv('date,value\n2024-01-01,10\n2024-01-02,20\n');
    const result = parseCSVData(fp, 'no-such-col');

    expect(result.dates).toEqual(['', '']);
    expect(result.series.get('value')).toBeDefined();
  });

  it('trims whitespace from CSV values', () => {
    const fp = writeTempCsv('date,value\n  2024-01-01  ,  10  \n  2024-01-02  ,  20  \n');
    const result = parseCSVData(fp, 'date');

    expect(result.dates).toEqual(['2024-01-01', '2024-01-02']);
    expect(Array.from(result.series.get('value')!)).toEqual([10, 20]);
  });
});

// ---------------------------------------------------------------------------
// outputCSV
// ---------------------------------------------------------------------------

describe('outputCSV', () => {
  it('writes CSV to a file with correct structure', () => {
    const outPath = path.join(tmpDir, 'output.csv');
    const result = makeForecastResult();
    const opts = makeOptions({ outputPath: outPath });

    outputCSV(result, ['series_a'], opts);

    const content = fs.readFileSync(outPath, 'utf-8');
    const lines = content.trim().split('\n');
    expect(lines[0]).toBe('series_id,horizon_step,point_forecast,q10,q50,q90');
    expect(lines.length).toBe(4);

    const fields1 = lines[1].split(',');
    expect(fields1[0]).toBe('series_a');
    expect(fields1[1]).toBe('1');
    expect(Number.parseFloat(fields1[2])).toBeCloseTo(10.1);
    expect(Number.parseFloat(fields1[3])).toBeCloseTo(2);
    expect(Number.parseFloat(fields1[4])).toBeCloseTo(6);
    expect(Number.parseFloat(fields1[5])).toBeCloseTo(10);
  });

  it('writes to stdout when no outputPath is set', () => {
    const writeSpy = vi.spyOn(process.stdout, 'write').mockImplementation(() => true);
    const result = makeForecastResult();
    const opts = makeOptions({ outputPath: undefined });

    outputCSV(result, ['s1'], opts);

    expect(writeSpy).toHaveBeenCalledOnce();
    const csv = writeSpy.mock.calls[0][0] as string;
    expect(csv).toContain('series_id,horizon_step,point_forecast,q10,q50,q90');
    expect(csv).toContain('s1,1,');
    expect(csv).toContain('s1,3,');

    writeSpy.mockRestore();
  });

  it('handles multiple series', () => {
    const outPath = path.join(tmpDir, 'output-multi.csv');
    const result = {
      pointForecast: [new Float32Array([1, 2]), new Float32Array([10, 20])],
      quantileForecast: [
        Array.from({ length: 10 }, () => new Float32Array([0, 0])),
        Array.from({ length: 10 }, () => new Float32Array([0, 0])),
      ],
    };
    const opts = makeOptions({ outputPath: outPath });

    outputCSV(result, ['first', 'second'], opts);

    const content = fs.readFileSync(outPath, 'utf-8');
    const lines = content.trim().split('\n');
    expect(lines.length).toBe(5);
    expect(lines[1].split(',')[0]).toBe('first');
    expect(lines[2].split(',')[0]).toBe('first');
    expect(lines[3].split(',')[0]).toBe('second');
    expect(lines[4].split(',')[0]).toBe('second');
  });
});

// ---------------------------------------------------------------------------
// outputJSON
// ---------------------------------------------------------------------------

describe('outputJSON', () => {
  it('writes JSON to a file with correct structure', () => {
    const outPath = path.join(tmpDir, 'output.json');
    const result = makeForecastResult();
    const opts = makeOptions({ outputPath: outPath, outputFormat: 'json' });

    outputJSON(result, ['series_a'], opts);

    const content = fs.readFileSync(outPath, 'utf-8');
    const json = JSON.parse(content);

    expect(json.model).toBe('timesfm-2.5');
    expect(json.horizon).toBe(3);
    expect(json.series).toHaveProperty('series_a');
    expect(json.series.series_a.point_forecast).toEqual([
      10.100000381469727, 20.200000762939453, 30.299999237060547,
    ]);
    expect(json.series.series_a.lower_80.length).toBe(3);
    expect(json.series.series_a.upper_80.length).toBe(3);
    expect(json.series.series_a.quantiles).toHaveProperty('q10');
    expect(json.series.series_a.quantiles).toHaveProperty('q50');
    expect(json.series.series_a.quantiles).toHaveProperty('q90');
  });

  it('writes to stdout when no outputPath is set', () => {
    const writeSpy = vi.spyOn(process.stdout, 'write').mockImplementation(() => true);
    const result = makeForecastResult();
    const opts = makeOptions({ outputPath: undefined, outputFormat: 'json' });

    outputJSON(result, ['s1'], opts);

    expect(writeSpy).toHaveBeenCalledOnce();
    const json = JSON.parse(writeSpy.mock.calls[0][0] as string);
    expect(json.model).toBe('timesfm-2.5');
    expect(json.series.s1.point_forecast).toHaveLength(3);

    writeSpy.mockRestore();
  });

  it('handles multiple series', () => {
    const outPath = path.join(tmpDir, 'output-multi.json');
    const result = {
      pointForecast: [new Float32Array([1, 2]), new Float32Array([10, 20])],
      quantileForecast: [
        Array.from({ length: 10 }, () => new Float32Array([0, 0])),
        Array.from({ length: 10 }, () => new Float32Array([0, 0])),
      ],
    };
    const opts = makeOptions({ outputPath: outPath, outputFormat: 'json' });

    outputJSON(result, ['first', 'second'], opts);

    const content = fs.readFileSync(outPath, 'utf-8');
    const json = JSON.parse(content);

    expect(json.series).toHaveProperty('first');
    expect(json.series).toHaveProperty('second');
    expect(json.series.first.point_forecast).toEqual([1, 2]);
    expect(json.series.second.point_forecast).toEqual([10, 20]);
  });

  it('includes all 10 quantile levels', () => {
    const outPath = path.join(tmpDir, 'output-quantiles.json');
    const result = makeForecastResult();
    const opts = makeOptions({ outputPath: outPath, outputFormat: 'json' });

    outputJSON(result, ['qtest'], opts);

    const content = fs.readFileSync(outPath, 'utf-8');
    const json = JSON.parse(content);
    const qt = json.series.qtest.quantiles;

    for (let i = 1; i <= 9; i++) {
      expect(qt).toHaveProperty(`q${i}0`);
      expect(qt[`q${i}0`]).toHaveLength(3);
    }
  });
});

// ---------------------------------------------------------------------------
// csvForecast (mock model — pure logic coverage)
// ---------------------------------------------------------------------------

describe('csvForecast', () => {
  it('forecasts a CSV and outputs CSV to file', async () => {
    const inPath = writeTempCsv('date,value\n2024-01-01,10\n2024-01-02,20\n2024-01-03,30\n');
    const outPath = path.join(tmpDir, 'fc-out.csv');

    mockForecast.mockResolvedValue({
      pointForecast: [new Float32Array([1.5, 2.5])],
      quantileForecast: [Array.from({ length: 10 }, () => new Float32Array([0.5, 1.5]))],
    });

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputPath: outPath,
        outputFormat: 'csv',
        horizon: 2,
      }),
    );

    expect(mockFromPretrained).toHaveBeenCalledWith({ modelPath: '/fake/model' });
    expect(mockCompile).toHaveBeenCalled();
    expect(mockForecast).toHaveBeenCalledWith(2, expect.any(Array));
    expect(mockDispose).toHaveBeenCalled();

    const content = fs.readFileSync(outPath, 'utf-8');
    expect(content).toContain('series_id,horizon_step,point_forecast');
    expect(content).toContain('value');
  });

  it('forecasts a CSV and outputs JSON to file', async () => {
    const inPath = writeTempCsv('date,a,b\n2024-01-01,1,10\n2024-01-02,2,20\n');
    const outPath = path.join(tmpDir, 'fc-out.json');

    mockForecast.mockResolvedValue({
      pointForecast: [new Float32Array([10]), new Float32Array([100])],
      quantileForecast: [
        Array.from({ length: 10 }, () => new Float32Array([1])),
        Array.from({ length: 10 }, () => new Float32Array([2])),
      ],
    });

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputPath: outPath,
        outputFormat: 'json',
        horizon: 1,
      }),
    );

    const content = fs.readFileSync(outPath, 'utf-8');
    const json = JSON.parse(content);

    expect(json.model).toBe('timesfm-2.5');
    expect(json.horizon).toBe(1);
    expect(json.series).toHaveProperty('a');
    expect(json.series).toHaveProperty('b');
    expect(json.series.a.point_forecast).toEqual([10]);
    expect(json.series.b.point_forecast).toEqual([100]);
  });

  it('outputs JSON to stdout when no outputPath', async () => {
    const inPath = writeTempCsv('date,x\n2024-01-01,5\n');
    const writeSpy = vi.spyOn(process.stdout, 'write').mockImplementation(() => true);

    mockForecast.mockResolvedValue({
      pointForecast: [new Float32Array([99])],
      quantileForecast: [Array.from({ length: 10 }, () => new Float32Array([0]))],
    });

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputFormat: 'json',
        horizon: 1,
      }),
    );

    const json = JSON.parse(writeSpy.mock.calls[0][0] as string);
    expect(json.series.x.point_forecast).toEqual([99]);

    writeSpy.mockRestore();
  });

  it('outputs CSV to stdout when no outputPath', async () => {
    const inPath = writeTempCsv('date,y\n2024-01-01,7\n');
    const writeSpy = vi.spyOn(process.stdout, 'write').mockImplementation(() => true);

    mockForecast.mockResolvedValue({
      pointForecast: [new Float32Array([42])],
      quantileForecast: [Array.from({ length: 10 }, () => new Float32Array([0]))],
    });

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputFormat: 'csv',
        horizon: 1,
      }),
    );

    expect(writeSpy).toHaveBeenCalled();
    const csv = writeSpy.mock.calls[0][0] as string;
    expect(csv).toContain('y,1,42');

    writeSpy.mockRestore();
  });

  it('passes model configuration options to createForecastConfig', async () => {
    const inPath = writeTempCsv('date,v\n2024-01-01,1\n');

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputFormat: 'json',
        horizon: 4,
        maxContext: 256,
        normalizeInputs: true,
        forceFlipInvariance: true,
        inferIsPositive: true,
        fixQuantileCrossing: true,
        useContinuousQuantileHead: true,
      }),
    );

    expect(mockCreateForecastConfig).toHaveBeenCalledWith({
      maxContext: 256,
      maxHorizon: 4,
      normalizeInputs: true,
      forceFlipInvariance: true,
      inferIsPositive: true,
      fixQuantileCrossing: true,
      useContinuousQuantileHead: true,
    });
  });

  it('passes correct series data to model.forecast', async () => {
    const inPath = writeTempCsv('date,one,two\n2024-01-01,1,10\n2024-01-02,2,20\n');

    await csvForecast(
      makeOptions({
        inputPath: inPath,
        outputFormat: 'json',
        horizon: 2,
      }),
    );

    const forecastCall = mockForecast.mock.calls[0];
    expect(forecastCall[0]).toBe(2); // horizon
    const inputs = forecastCall[1] as Float32Array[];
    expect(inputs.length).toBe(2);
    expect(Array.from(inputs[0])).toEqual([1, 2]);
    expect(Array.from(inputs[1])).toEqual([10, 20]);
  });
});
