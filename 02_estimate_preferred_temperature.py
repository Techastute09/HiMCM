#!/usr/bin/env python3
"""
Estimate population preferred SET from ASHRAE thermal_preference votes.

Model:
  Multinomial logistic regression using a quadratic polynomial of observed SET.
  Classes: cooler, no change, warmer.

Preferred SET is defined operationally as the SET value maximizing:
  P(thermal_preference == "no change" | SET)

This is a population-level statistical preference proxy, not an individual's
ideal temperature and not a guarantee of thermal comfort.

Outputs:
  - preferred_temperature_summary.csv
  - preference_model_coefficients.csv
  - preference_probability_curve.csv
  - preference_metrics.csv
  - preferred_temperature_model.joblib
  - preference_probability_plot.png
  - run_summary.txt
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


CLASSES = ["cooler", "no change", "warmer"]


def normalize_preference(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    aliases = {
        "cooler": "cooler",
        "no change": "no change",
        "no_change": "no change",
        "no-change": "no change",
        "warmer": "warmer",
    }
    return s.map(aliases)


def load_data(csv_path: Path, set_min: float, set_max: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = ["set", "thermal_preference"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["set"] = pd.to_numeric(df["set"], errors="coerce")
    df["preference_clean"] = normalize_preference(df["thermal_preference"])
    df = df.dropna(subset=["set", "preference_clean"]).copy()
    df = df[df["set"].between(set_min, set_max)]
    df = df[df["preference_clean"].isin(CLASSES)]
    if df.empty:
        raise ValueError("No valid SET/preference records remain after cleaning.")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Estimate preferred SET from thermal preference.")
    parser.add_argument("--input", default="raw-data/db_measurements_v2.0.1.csv")
    parser.add_argument("--output-dir", default="preferred_temperature_output")
    parser.add_argument("--set-min", type=float, default=10.0)
    parser.add_argument("--set-max", type=float, default=50.0)
    parser.add_argument("--interval-threshold", type=float, default=0.50,
                        help="No-change probability threshold used for interval.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.interval_threshold < 1:
        raise ValueError("--interval-threshold must be between 0 and 1.")

    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path, args.set_min, args.set_max)
    X = df[["set"]]
    y = df["preference_clean"]

    # A grouped split is used for an honest diagnostic; deployment model is
    # subsequently refitted using all eligible observations.
    if "building_id" in df.columns and df["building_id"].notna().all():
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20,
                                     random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["building_id"]))
        split_method = "GroupShuffleSplit by building_id"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.20,
            random_state=args.random_state, stratify=y
        )
        split_method = "Stratified random row split (building_id unavailable)"

    diagnostic_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=3000, class_weight=None, solver="lbfgs"
        )),
    ])
    diagnostic_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    test_prob = diagnostic_model.predict_proba(X.iloc[test_idx])
    test_pred = diagnostic_model.predict(X.iloc[test_idx])
    diagnostic_classes = diagnostic_model.named_steps["logistic"].classes_
    metrics = {
        "model": "quadratic_multinomial_logistic",
        "split_method": split_method,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": accuracy_score(y.iloc[test_idx], test_pred),
        "log_loss": log_loss(y.iloc[test_idx], test_prob, labels=diagnostic_classes),
    }
    pd.DataFrame([metrics]).to_csv(out / "preference_metrics.csv", index=False)

    # Fit final model on all eligible records.
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=3000, class_weight=None, solver="lbfgs"
        )),
    ])
    model.fit(X, y)

    set_grid = np.linspace(max(args.set_min, df["set"].min()),
                           min(args.set_max, df["set"].max()), 400)
    grid_frame = pd.DataFrame({"set": set_grid})
    probabilities = model.predict_proba(grid_frame)
    model_classes = list(model.named_steps["logistic"].classes_)
    curve = pd.DataFrame({"SET_C": set_grid})
    for cls in CLASSES:
        curve[f"P_{cls.replace(' ', '_')}"] = probabilities[:, model_classes.index(cls)]
    curve.to_csv(out / "preference_probability_curve.csv", index=False)

    no_change_col = "P_no_change"
    peak_index = int(curve[no_change_col].to_numpy().argmax())
    preferred_set = float(curve.iloc[peak_index]["SET_C"])
    peak_probability = float(curve.iloc[peak_index][no_change_col])

    # Continuous interval(s) where P(no change) >= threshold.
    mask = curve[no_change_col].to_numpy() >= args.interval_threshold
    intervals = []
    if mask.any():
        starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
        ends = np.where(mask & ~np.r_[mask[1:], False])[0]
        intervals = [
            (float(curve.iloc[s]["SET_C"]), float(curve.iloc[e]["SET_C"]))
            for s, e in zip(starts, ends)
        ]
    interval_text = "; ".join(f"{a:.2f}-{b:.2f} °C" for a, b in intervals) if intervals else "No interval at selected threshold"

    summary_df = pd.DataFrame([{
        "preferred_SET_C": preferred_set,
        "max_no_change_probability": peak_probability,
        "interval_threshold": args.interval_threshold,
        "no_change_interval_SET_C": interval_text,
        "n_valid_records": len(df),
        "n_cooler": int((y == "cooler").sum()),
        "n_no_change": int((y == "no change").sum()),
        "n_warmer": int((y == "warmer").sum()),
        "set_min_observed": float(df["set"].min()),
        "set_max_observed": float(df["set"].max()),
        "definition": "argmax P(no change | observed SET)",
    }])
    summary_df.to_csv(out / "preferred_temperature_summary.csv", index=False)

    # Export coefficients on the standardized polynomial feature scale.
    poly_names = model.named_steps["poly"].get_feature_names_out(["set"])
    logistic = model.named_steps["logistic"]
    coef_rows = []
    for i, cls in enumerate(logistic.classes_):
        for term, coef in zip(poly_names, logistic.coef_[i]):
            coef_rows.append({"class": cls, "term_standardized": term, "coefficient": coef})
        coef_rows.append({"class": cls, "term_standardized": "intercept", "coefficient": logistic.intercept_[i]})
    pd.DataFrame(coef_rows).to_csv(out / "preference_model_coefficients.csv", index=False)

    joblib.dump({
        "pipeline": model,
        "input_feature": "set",
        "classes": list(logistic.classes_),
        "preferred_SET_C": preferred_set,
        "definition": "argmax P(no change | SET)",
        "interval_threshold": args.interval_threshold,
        "valid_set_range": [float(df["set"].min()), float(df["set"].max())],
    }, out / "preferred_temperature_model.joblib")

    plt.figure(figsize=(8, 5))
    plt.plot(curve["SET_C"], curve["P_cooler"], label="Cooler")
    plt.plot(curve["SET_C"], curve["P_no_change"], label="No change")
    plt.plot(curve["SET_C"], curve["P_warmer"], label="Warmer")
    plt.axvline(preferred_set, linestyle="--", label=f"Preferred SET = {preferred_set:.2f}°C")
    plt.axhline(args.interval_threshold, linestyle=":", label=f"Interval threshold = {args.interval_threshold:.2f}")
    plt.xlabel("Observed SET (°C)")
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.title("Thermal preference probability model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "preference_probability_plot.png", dpi=180)
    plt.close()

    summary_lines = [
        "Preferred SET estimation from ASHRAE thermal_preference",
        f"Input: {input_path.resolve()}",
        f"Valid records: {len(df)}",
        f"Cooler: {(y == 'cooler').sum()}",
        f"No change: {(y == 'no change').sum()}",
        f"Warmer: {(y == 'warmer').sum()}",
        f"Preferred SET: {preferred_set:.4f} °C",
        f"Maximum P(no change): {peak_probability:.4f}",
        f"No-change interval at threshold {args.interval_threshold:.2f}: {interval_text}",
        f"Diagnostic split: {split_method}",
        f"Diagnostic accuracy: {metrics['accuracy']:.6f}",
        f"Diagnostic log loss: {metrics['log_loss']:.6f}",
        "",
        "Interpretation: population-level preference proxy, not an individual ideal temperature.",
    ]
    (out / "run_summary.txt").write_text("\\n".join(summary_lines), encoding="utf-8")
    print("\\n".join(summary_lines))
    print(f"\\nOutputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
