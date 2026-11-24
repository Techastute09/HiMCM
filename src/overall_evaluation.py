import numpy as np
import pandas as pd

from comfort_benefit import generate_population, expected_benefit
from cooling_cost import calculate_annual_cost
from  calculate_experienced_t import predict_experienced_temperature, calculate_device_temperatures

DEFAULT_DEVICES = {
    "Fan": {
        "temperature_col": "T_fan",
        "coverage_fraction": 0.40,
        "power_kw": 0.25,
        "capex": 500.0,
        "water_m3_per_year": 0.0,
        "maintenance_per_year": 40.0,
    },
    "Evaporative cooler": {
        "temperature_col": "T_evap",
        "coverage_fraction": 0.70,
        "power_kw": 1.20,
        "capex": 1800.0,
        "water_m3_per_year": 120.0,
        "maintenance_per_year": 120.0,
    },
    "Large evaporative cooler": {
        "temperature_col": "T_large",
        "coverage_fraction": 0.90,
        "power_kw": 2.50,
        "capex": 3500.0,
        "water_m3_per_year": 240.0,
        "maintenance_per_year": 200.0,
    },
}

def calculate_cooling_scenarios(ta, velocity, rh, tr):

    scenarios = {
        "Baseline": {
            "ta_h": ta,
            "vel_h": velocity,
            "rh": rh,
            "tr": tr
        },

        "Fan": {
            "ta_h": ta,
            "vel_h": velocity + 1.0,
            "rh": rh,
            "tr": tr
        },

        "Evaporative cooler": {
            "ta_h": ta - 2.0,
            "vel_h": velocity + 0.5,
            "rh": rh + 7.0,
            "tr": tr
        },

        "Large evaporative cooler": {
            "ta_h": ta - 4.0,
            "vel_h": velocity + 1.0,
            "rh": rh + 13.0,
            "tr": tr
        }
    }
    return scenarios

def evaluate_devices(
    df,
    dt_hours=0.25,
    operating_days_per_year=300,
    electricity_price=0.25,
    water_price=2.0,
    lifetime_years=10,
    discount_rate=0.05,
    devices=None,
    comfort_params=None,
):
    """
    Evaluate device alternatives.

    Comfort benefit unit:
      discomfort-degree-equivalent * passenger-hour

    Annualization assumes the supplied sample period is representative of
    one operating day and is repeated operating_days_per_year times.
    """
    required = {"passengers_waiting", "T_before"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    devices = devices or DEFAULT_DEVICES
    comfort_params = comfort_params or {}
    preferred, width = generate_population(**comfort_params)

    daily_hours = len(df) * dt_hours
    annual_hours = daily_hours * operating_days_per_year
    daily_benefits = {}

    rows = []
    for name, spec in devices.items():
        temp_col = spec["temperature_col"]
        print(f"Evaluating device: {name} (temperature column: {temp_col})")
        if temp_col not in df.columns:
            raise ValueError(f"Missing temperature column {temp_col!r}")

        covered_fraction = float(spec["coverage_fraction"])
        if not 0 <= covered_fraction <= 1:
            raise ValueError(f"Invalid coverage fraction for {name}")

        daily_benefit = 0.0
        for row in df.itertuples(index=False):
            ta_h = float(getattr(row, "ta_h"))
            vel_h = float(getattr(row, "vel_h"))
            rh = float(getattr(row, "rh"))
            tr = float(getattr(row, "tr"))

            env = calculate_cooling_scenarios(ta_h, vel_h, rh, tr)
            t_before = predict_experienced_temperature(ta_h, vel_h, rh, tr)
            t_after = predict_experienced_temperature(env[name]["ta_h"], env[name]["vel_h"], env[name]["rh"], env[name]["tr"])

            people = max(0.0, float(getattr(row, "passengers_waiting")))
            covered_people = people * covered_fraction
            benefit_per_person = expected_benefit(
                #float(getattr(row, "T_before")),
                #float(getattr(row, temp_col)),
                t_before,
                t_after,
                preferred,
                width,
            )
            daily_benefit += covered_people * benefit_per_person * dt_hours

        annual_benefit = daily_benefit * operating_days_per_year
        daily_benefits[name] = daily_benefit

        cost = calculate_annual_cost(
            power_kw=spec["power_kw"],
            operating_hours_per_year=annual_hours,
            electricity_price_per_kwh=electricity_price,
            capex=spec["capex"],
            lifetime_years=lifetime_years,
            discount_rate=discount_rate,
            water_m3_per_year=spec["water_m3_per_year"],
            water_price_per_m3=water_price,
            maintenance_per_year=spec["maintenance_per_year"],
        )

        energy = cost["annual_energy_kwh"]
        annual_cost = cost["annual_total_cost"]

        rows.append({
            "Device": name,
            "Daily_Comfort_Benefit": daily_benefit,
            "Annual_Comfort_Benefit": annual_benefit,
            **cost,
            "Benefit_per_kWh": annual_benefit / energy if energy else np.nan,
            "Benefit_per_Dollar": annual_benefit / annual_cost if annual_cost else np.nan,
        })

    result = pd.DataFrame(rows)

    # Min-max normalization. If all values are equal, assign 1 to all.
    metric_cols = [
        "Annual_Comfort_Benefit",
        "Benefit_per_kWh",
        "Benefit_per_Dollar",
    ]
    norm = pd.DataFrame(index=result.index)
    for col in metric_cols:
        low, high = result[col].min(), result[col].max()
        norm[col] = 1.0 if np.isclose(high, low) else (
            (result[col] - low) / (high - low)
        )

    result["Overall_Score"] = (
        0.50 * norm["Annual_Comfort_Benefit"]
        + 0.25 * norm["Benefit_per_kWh"]
        + 0.25 * norm["Benefit_per_Dollar"]
    )
    result = result.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
    return result
