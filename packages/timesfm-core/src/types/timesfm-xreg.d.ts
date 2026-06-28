/**
 * Type declaration for the optional @agentix-e/timesfm-xreg peer dependency.
 *
 * This module is dynamically imported by TimesFMModel.forecastWithCovariates()
 * only when the package is installed.  When not installed, a descriptive
 * error message guides the user to install it.
 *
 * This declaration eliminates the `@ts-ignore` escape hatch while keeping
 * the dependency truly optional — no compile-time errors for consumers who
 * don't need covariate forecasting.
 */
declare module '@agentix-e/timesfm-xreg' {
  import type {
    TimesFMModel,
    CovariateForecastParams,
    CovariateForecastOutput,
  } from '@agentix-e/timesfm-core';

  /**
   * Run TimesFM forecast with exogenous covariates.
   *
   * @param model  A compiled TimesFMModel instance.
   * @param params Covariate forecast parameters.
   */
  export function forecastWithCovariates(
    model: TimesFMModel,
    params: CovariateForecastParams,
  ): Promise<CovariateForecastOutput>;

  /** Standalone scikit-learn compatible OneHotEncoder. */
  export class OneHotEncoder {
    constructor(options?: { drop?: 'first' | 'none'; handleUnknown?: 'ignore' | 'error' });
    fit(categories: Array<string | number>): void;
    transform(categories: Array<string | number>): number[][];
    readonly numColumns: number;
  }
}
