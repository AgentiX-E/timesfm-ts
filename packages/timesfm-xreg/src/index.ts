/**
 * @agentix/timesfm-xreg
 *
 * Exogenous covariates (XReg) extension for TimesFM.
 *
 * Enables forecast_with_covariates() with support for:
 *   - Dynamic numerical covariates
 *   - Dynamic categorical covariates
 *   - Static numerical covariates
 *   - Static categorical covariates
 *
 * Two modes:
 *   - "xreg + timesfm": Fit covariates → forecast residuals → combine.
 *   - "timesfm + xreg": Forecast → fit covariates on residuals → combine.
 */

export { forecastWithCovariates } from './xreg-engine';
export type { CovariateForecastParams, CovariateForecastOutput, XRegMode } from './xreg-engine';

export { OneHotEncoder } from './one-hot-encoder';
export type { OneHotEncoderState, Category } from './one-hot-encoder';
