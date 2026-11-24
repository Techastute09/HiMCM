#!/usr/bin/env python3
"""
Four-variable quadratic regression for ASHRAE SET.

Inputs:
  - CSV containing ta, tr, rh, vel, set
  - building_id is recommended for group-wise train/test splitting

Outputs (under --output-dir):
  - regression_coefficients.csv
  - regression_metrics.csv
  - set_quadratic_model.joblib
  - test_predictions.csv
  - feature_importance_coefficients.csv
  - set_regression_plot.png
  - run_summary.txt

The target is the database's SET field (Standard Effective Temperature),
not a directly measured personal "experienced temperature".
"""

import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import PolynomialFeatures


FEATURES = ["ta", "tr", "rh", "vel"]
TARGET = "set"
RANGES = {
    "ta": (5, 45),
    "tr": (5, 60),
    "rh": (0, 100),
    "vel": (0, 5),
    "set": (10, 50),
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()

    for col, (low, high) in RANGES.items():
        df = df[df[col].between(low, high)]
    if df.empty:
        raise ValueError("No rows remain after cleaning; check column names and ranges.")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fit four-variable quadratic regression for SET.")
    parser.add_argument("--input", default="raw-data/db_measurements_v2.0.1.csv",
                        help="Input ASHRAE CSV path.")
    parser.add_argument("--output-dir", default="linear_results",
                        help="Directory for model and result files.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    X = df[FEATURES]
    y = df[TARGET]

    if "building_id" in df.columns and df["building_id"].notna().all():
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                     random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["building_id"]))
        split_method = "GroupShuffleSplit by building_id"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=args.test_size,
            random_state=args.random_state
        )
        split_method = "Random row split (building_id unavailable)"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    reg = LinearRegression()
    reg.fit(X_train_poly, y_train)
    pred = reg.predict(X_test_poly)

    metrics = {
        "model": "four_variable_quadratic",
        "target": "SET",
        "split_method": split_method,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }
    pd.DataFrame([metrics]).to_csv(out / "regression_metrics.csv", index=False)

    # Refit on all valid observations for later deployment.
    X_all_poly = poly.fit_transform(X)
    reg_all = LinearRegression()
    reg_all.fit(X_all_poly, y)
    feature_names = poly.get_feature_names_out(FEATURES)

    coef_table = pd.DataFrame({
        "term": ["intercept"] + list(feature_names),
        "coefficient": [reg_all.intercept_] + list(reg_all.coef_),
    })
    coef_table.to_csv(out / "regression_coefficients.csv", index=False)

    pd.DataFrame({
        "term": feature_names,
        "coefficient": reg_all.coef_,
        "absolute_coefficient": np.abs(reg_all.coef_),
    }).sort_values("absolute_coefficient", ascending=False).to_csv(
        out / "feature_importance_coefficients.csv", index=False
    )

    # Store the fitted feature expansion and full-data regression together.
    joblib.dump({
        "polynomial_features": poly,
        "regressor": reg_all,
        "input_features": FEATURES,
        "target": TARGET,
        "target_interpretation": "ASHRAE SET, not directly measured experienced temperature",
        "valid_ranges": RANGES,
    }, out / "set_quadratic_model.joblib")

    pd.DataFrame({
        "actual_SET": y_test.to_numpy(),
        "predicted_SET": pred,
        "residual": y_test.to_numpy() - pred,
    }).to_csv(out / "test_predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, s=9, alpha=0.35)
    lo = min(float(y_test.min()), float(pred.min()))
    hi = max(float(y_test.max()), float(pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual SET (°C)")
    plt.ylabel("Predicted SET (°C)")
    plt.title("Four-variable quadratic SET regression")
    plt.tight_layout()
    plt.savefig(out / "set_regression_plot.png", dpi=180)
    plt.close()

    summary = [
        "Four-variable quadratic SET regression",
        f"Input: {input_path.resolve()}",
        f"Rows after cleaning: {len(df)}",
        f"Split: {split_method}",
        f"Train rows: {len(train_idx)}",
        f"Test rows: {len(test_idx)}",
        f"Test MAE: {metrics['MAE']:.6f}",
        f"Test RMSE: {metrics['RMSE']:.6f}",
        f"Test R2: {metrics['R2']:.6f}",
        "",
        "Deployment model: refitted on all cleaned rows.",
        "Target is ASHRAE SET, not a directly measured personal experienced temperature.",
        "Coefficient terms use raw input units and are listed in regression_coefficients.csv.",
    ]
    (out / "run_summary.txt").write_text("\\n".join(summary), encoding="utf-8")
    print("\\n".join(summary))
    print(f"\\nOutputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
