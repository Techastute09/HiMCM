"""
calculate_experienced_t.py

Read the latest four-variable quadratic regression coefficients from
regression_coefficients.csv and calculate predicted SET.

The project uses predicted SET as its Experienced Temperature Proxy.

Expected coefficient CSV columns:
    term, coefficient

Expected terms:
    intercept, ta, tr, rh, vel,
    ta^2, tr^2, rh^2, vel^2,
    ta tr, ta rh, ta vel, tr rh, tr vel, rh vel
"""

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------------------------------
# 1. Locate and load regression coefficients
# -----------------------------------------------------

# Put regression_coefficients.csv in the same folder as this Python file,
# or change this path to your project's actual output location.
COEFFICIENT_FILE = (
    Path(__file__).resolve().parent.parent
    / "linear_results/regression_coefficients.csv"
)


def load_regression_coefficients(coefficient_file=COEFFICIENT_FILE):
    """Load the polynomial regression coefficients from CSV."""

    coefficient_file = Path(coefficient_file)

    if not coefficient_file.exists():
        raise FileNotFoundError(
            f"Regression coefficient file not found: {coefficient_file}"
        )

    df = pd.read_csv(coefficient_file)

    required_columns = {"term", "coefficient"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "regression_coefficients.csv must contain "
            "columns: term, coefficient"
        )

    df["term"] = df["term"].astype(str).str.strip()
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="raise")

    if df["term"].duplicated().any():
        raise ValueError("Duplicate terms found in regression_coefficients.csv.")

    coefficients = dict(zip(df["term"], df["coefficient"]))

    required_terms = {
        "intercept",
        "ta", "tr", "rh", "vel",
        "ta^2", "tr^2", "rh^2", "vel^2",
        "ta tr", "ta rh", "ta vel",
        "tr rh", "tr vel", "rh vel",
    }

    missing = required_terms - set(coefficients)
    if missing:
        raise ValueError(
            f"Missing regression terms in coefficient file: {sorted(missing)}"
        )

    print(f"Regression coefficients loaded: {coefficient_file}")
    return coefficients


# Load coefficients once when this module is imported.
BETA = load_regression_coefficients()


# -----------------------------------------------------
# 2. Calculate experienced temperature proxy (SET)
# -----------------------------------------------------

def predict_experienced_temperature(
    ta_h,
    vel_h,
    rh,
    tr,
    coefficients=None
):
    """
    Calculate predicted SET using the full quadratic equation.

    Inputs:
        ta_h  : air temperature (°C)
        vel_h : air velocity (m/s)
        rh    : relative humidity (%)
        tr    : mean radiant temperature (°C)

    Inputs can be scalars or NumPy-compatible arrays.
    """

    if coefficients is None:
        coefficients = BETA

    ta = np.asarray(ta_h, dtype=float)
    tr = np.asarray(tr, dtype=float)
    rh = np.asarray(rh, dtype=float)
    vel = np.asarray(vel_h, dtype=float)

    result = (
        coefficients["intercept"]
        + coefficients["ta"] * ta
        + coefficients["tr"] * tr
        + coefficients["rh"] * rh
        + coefficients["vel"] * vel
        + coefficients["ta^2"] * ta**2
        + coefficients["tr^2"] * tr**2
        + coefficients["rh^2"] * rh**2
        + coefficients["vel^2"] * vel**2
        + coefficients["ta tr"] * ta * tr
        + coefficients["ta rh"] * ta * rh
        + coefficients["ta vel"] * ta * vel
        + coefficients["tr rh"] * tr * rh
        + coefficients["tr vel"] * tr * vel
        + coefficients["rh vel"] * rh * vel
    )

    result = np.clip(result, 10.0, 50.0)
    if np.ndim(result) == 0:
        return float(result)

    return np.asarray(result)


# -----------------------------------------------------
# 3. Calculate one environment or multiple scenarios
# -----------------------------------------------------

def calculate_device_temperatures(env4):
    """
    Calculate predicted SET for one environment.

    Required keys:
        ta_h, vel_h, rh, tr

    Returns:
        {"Env4": predicted_SET}
    """

    result = predict_experienced_temperature(
        ta_h=env4["ta_h"],
        vel_h=env4["vel_h"],
        rh=env4["rh"],
        tr=env4["tr"],
    )

    return {"Env4": result}


def calculate_scenario_temperatures(scenarios):
    """
    Optional helper for multiple device scenarios.

    Example:
        scenarios = {
            "Baseline": {"ta_h": 35, "vel_h": 0.3, "rh": 50, "tr": 42},
            "Fan":      {"ta_h": 35, "vel_h": 1.5, "rh": 50, "tr": 42},
        }
    """

    results = {}

    for name, env in scenarios.items():
        results[name] = predict_experienced_temperature(
            ta_h=env["ta_h"],
            vel_h=env["vel_h"],
            rh=env["rh"],
            tr=env["tr"],
        )

    return results


if __name__ == "__main__":
    # Simple standalone test example.
    example = {
        "ta_h": 35,
        "tr": 42,
        "rh": 50,
        "vel_h": 1.5,
    }

    predicted = predict_experienced_temperature(**example)
    print(f"Predicted SET / Experienced Temperature Proxy: {predicted:.4f} °C")
