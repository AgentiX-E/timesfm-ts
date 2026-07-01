#!/usr/bin/env python3
"""
Generate golden reference outputs from Python TimesFM 2.5 for cross-validation.

This script runs the official TimesFM Python reference implementation on
a set of fixed inputs and saves the forecast outputs as JSON fixtures.
The TypeScript test suite loads these fixtures and validates that the
TypeScript reimplementation produces numerically equivalent results.

Usage:
    python scripts/generate-golden-outputs.py \
        --output packages/timesfm-core/test/golden-outputs.json
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_simple_trend():
    return np.linspace(10, 100, 256, dtype=np.float32)


def generate_sine_wave():
    t = np.linspace(0, 8 * np.pi, 512, dtype=np.float32)
    return np.sin(t) + 0.1 * np.random.RandomState(42).randn(512).astype(np.float32)


def generate_step_function():
    data = np.ones(300, dtype=np.float32) * 10
    data[150:] = 20
    return data


def generate_with_nans():
    rng = np.random.RandomState(99)
    data = rng.randn(256).astype(np.float32) * 5 + 50
    data[50:55] = np.nan
    data[200:205] = np.nan
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="packages/timesfm-core/test/golden-outputs.json")
    args = parser.parse_args()

    from timesfm import TimesFM_2p5_200M_torch
    from timesfm.configs import ForecastConfig as PyForecastConfig

    print("Loading TimesFM 2.5 200M from HuggingFace...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )

    print("Compiling model...")
    tfm.compile(
        PyForecastConfig(
            max_context=512,
            max_horizon=64,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    test_cases = {
        "simple_trend": {
            "description": "Linear trend 10-100 over 256 points",
            "data": generate_simple_trend().tolist(),
            "horizon": 24,
        },
        "sine_wave": {
            "description": "Sine wave with small noise over 512 points",
            "data": generate_sine_wave().tolist(),
            "horizon": 32,
        },
        "step_function": {
            "description": "Step function - abrupt regime change at midpoint",
            "data": generate_step_function().tolist(),
            "horizon": 24,
        },
        "with_nans": {
            "description": "Series with internal NaN gaps",
            "data": generate_with_nans().tolist(),
            "horizon": 24,
        },
    }

    results = {}
    for name, case in test_cases.items():
        print(f"\n--- {name}: {case['description']} ---")
        data = np.array(case["data"], dtype=np.float32)
        horizon = case["horizon"]

        point_fc, quantile_fc = tfm.forecast(horizon, [data])

        results[name] = {
            "description": case["description"],
            "input_length": int(len(data)),
            "horizon": horizon,
            "point_forecast": point_fc[0, :].tolist(),
            "quantile_forecast": quantile_fc[0, :, :].tolist(),
            "input_summary": {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "nan_count": int(np.isnan(data).sum()),
            },
        }
        print(f"  Generated {horizon}-step forecast (point + quantiles)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "model": "google/timesfm-2.5-200m-pytorch",
                "source": "Python TimesFM reference implementation",
                "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "test_cases": results,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Golden outputs written to {output_path}")


if __name__ == "__main__":
    main()
