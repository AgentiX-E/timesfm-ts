/**
 * CSV forecasting engine for the TimesFM CLI.
 *
 * Reads a CSV file, extracts time series, runs TimesFM forecasts,
 * and outputs results as CSV or JSON.
 */

import * as fs from 'node:fs';
import { parse } from 'csv-parse/sync';
import { stringify } from 'csv-stringify/sync';
import { TimesFMModel, createForecastConfig } from '@agentix/timesfm-core';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CSVForecastOptions {
  inputPath: string;
  horizon: number;
  modelPath: string;
  dateCol: string;
  valueCols?: string[];
  outputPath?: string;
  outputFormat: 'csv' | 'json';
  maxContext: number;
  normalizeInputs: boolean;
  forceFlipInvariance: boolean;
  inferIsPositive: boolean;
  fixQuantileCrossing: boolean;
  useContinuousQuantileHead: boolean;
}

interface ParsedCSV {
  dates: string[];
  series: Map<string, Float32Array>;
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

export function parseCSVData(filePath: string, dateCol: string, valueCols?: string[]): ParsedCSV {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const records: Record<string, string>[] = parse(raw, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });

  if (records.length === 0) {
    throw new Error(`Empty CSV file: ${filePath}`);
  }

  // Identify columns
  const allCols = Object.keys(records[0]);
  const numericCols = valueCols ?? allCols.filter((c) => c !== dateCol);

  // Extract dates
  const dates = records.map((r) => r[dateCol] ?? '');

  // Extract series
  const series = new Map<string, Float32Array>();
  for (const col of numericCols) {
    const values: number[] = [];
    for (const record of records) {
      const v = parseFloat(record[col]);
      values.push(Number.isFinite(v) ? v : NaN);
    }
    series.set(col, removeTrailingNaN(new Float32Array(values)));
  }

  return { dates, series };
}

export function removeTrailingNaN(arr: Float32Array): Float32Array {
  let end = arr.length;
  while (end > 0 && Number.isNaN(arr[end - 1])) end--;
  return arr.slice(0, end);
}

// ---------------------------------------------------------------------------
// Main forecast function
// ---------------------------------------------------------------------------

export async function csvForecast(options: CSVForecastOptions): Promise<void> {
  // Parse input
  const { series } = parseCSVData(options.inputPath, options.dateCol, options.valueCols);

  console.error(`Loaded ${series.size} series from ${options.inputPath}`);

  // Create model
  const model = await TimesFMModel.fromPretrained({
    modelPath: options.modelPath,
  });

  // Compile
  const fc = createForecastConfig({
    maxContext: options.maxContext,
    maxHorizon: options.horizon,
    normalizeInputs: options.normalizeInputs,
    forceFlipInvariance: options.forceFlipInvariance,
    inferIsPositive: options.inferIsPositive,
    fixQuantileCrossing: options.fixQuantileCrossing,
    useContinuousQuantileHead: options.useContinuousQuantileHead,
  });

  model.compile(fc);
  console.error(`Compiled model with maxContext=${fc.maxContext}, maxHorizon=${fc.maxHorizon}`);

  // Forecast
  const inputList: Float32Array[] = [];
  const seriesNames: string[] = [];

  for (const [name, data] of series) {
    inputList.push(data);
    seriesNames.push(name);
  }

  console.error(`Forecasting ${inputList.length} series for ${options.horizon} steps...`);
  const result = await model.forecast(options.horizon, inputList);

  // Output
  if (options.outputFormat === 'json') {
    outputJSON(result, seriesNames, options);
  } else {
    outputCSV(result, seriesNames, options);
  }

  await model.dispose();
  console.error('Done.');
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

export function outputCSV(
  result: { pointForecast: Float32Array[]; quantileForecast: Float32Array[][] },
  seriesNames: string[],
  options: CSVForecastOptions,
): void {
  const rows: Record<string, string | number>[] = [];

  for (let s = 0; s < result.pointForecast.length; s++) {
    const pf = result.pointForecast[s];
    const qf = result.quantileForecast[s];

    for (let h = 0; h < pf.length; h++) {
      rows.push({
        series_id: seriesNames[s],
        horizon_step: h + 1,
        point_forecast: pf[h],
        q10: qf[1][h],
        q50: qf[5][h],
        q90: qf[9][h],
      });
    }
  }

  const csv = stringify(rows, { header: true });

  if (options.outputPath) {
    fs.writeFileSync(options.outputPath, csv);
    console.error(`Wrote ${rows.length} rows to ${options.outputPath}`);
  } else {
    process.stdout.write(csv);
  }
}

export function outputJSON(
  result: { pointForecast: Float32Array[]; quantileForecast: Float32Array[][] },
  seriesNames: string[],
  options: CSVForecastOptions,
): void {
  const output: Record<string, unknown> = {
    model: 'timesfm-2.5',
    horizon: options.horizon,
    series: {},
  };

  const seriesOut = output.series as Record<string, unknown>;
  for (let s = 0; s < result.pointForecast.length; s++) {
    seriesOut[seriesNames[s]] = {
      point_forecast: Array.from(result.pointForecast[s]),
      lower_80: Array.from(result.quantileForecast[s][1]), // q10
      upper_80: Array.from(result.quantileForecast[s][9]), // q90
      quantiles: {
        q10: Array.from(result.quantileForecast[s][1]),
        q20: Array.from(result.quantileForecast[s][2]),
        q30: Array.from(result.quantileForecast[s][3]),
        q40: Array.from(result.quantileForecast[s][4]),
        q50: Array.from(result.quantileForecast[s][5]),
        q60: Array.from(result.quantileForecast[s][6]),
        q70: Array.from(result.quantileForecast[s][7]),
        q80: Array.from(result.quantileForecast[s][8]),
        q90: Array.from(result.quantileForecast[s][9]),
      },
    };
  }

  const json = JSON.stringify(output, null, 2);

  if (options.outputPath) {
    fs.writeFileSync(options.outputPath, json);
    console.error(`Wrote JSON to ${options.outputPath}`);
  } else {
    process.stdout.write(json);
  }
}
