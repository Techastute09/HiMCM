"""Scenario-based comfort-benefit and life-cycle cost evaluation for Option 2."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils_option2 import load_config, load_inputs, validate_inputs
from spatial_model_option2 import spatial_influence, outdoor_exchange
from passenger_flow_option2 import advance_passengers
from microclimate_option2 import get_time_row, get_radiant_map, integrate_environment
from comfort_model_option2 import (load_regression_coefficients, experienced_temperature,
                                   comfort_population, comfort_fraction)
from spatial_geometry_option2 import (generate_zone_geometry, assign_devices_to_zones,
                                      plot_station_geometry, validate_station_geometry)
from plots_option2 import make_plots


def _scenario_devices(devices, scenario):
    types = devices.device_type.astype(str).str.lower()
    if scenario == "Baseline":
        return devices.iloc[0:0].copy()
    if scenario == "Fan only":
        return devices[types.eq("fan")].copy()
    if scenario == "Evaporative only":
        return devices[types.isin(["evaporative", "evaporative_cooler"])].copy()
    if scenario == "Combined cooling":
        return devices.copy()
    raise ValueError(f"Unknown scenario: {scenario}")


def simulate_scenario(scenario, cfg, data, zones, devices, timestamps, coef, seed):
    """Run a scenario with identical exogenous inputs and random seed."""
    selected = _scenario_devices(devices, scenario)
    K, alpha = spatial_influence(zones, selected)
    zone_ids = zones.zone_id.astype(str).tolist()
    T = {str(r.zone_id): float(r.initial_temp_c) for _, r in zones.iterrows()}
    N = {str(r.zone_id): float(r.initial_people) for _, r in zones.iterrows()}
    w = float(cfg["simulation"]["initial_humidity_ratio"])
    pressure = float(cfg["simulation"]["pressure_pa"])
    dt_min = float(cfg["simulation"]["interval_minutes"])
    dt_hours = dt_min / 60.0
    rng = np.random.default_rng(seed)  # paired preference draws across scenarios
    records = []

    for timestamp in timestamps:
        weather = get_time_row(data["weather"], timestamp)
        arrivals = {z: 0.0 for z in zone_ids}
        arows = data["arrivals"][data["arrivals"].timestamp == timestamp]
        for _, row in arows.iterrows():
            arrivals[str(row.zone_id)] += float(row.arrivals)
        N_next, N_mean, exits = advance_passengers(N, zones, arrivals, data["transitions"], dt_hours)
        q_out = outdoor_exchange(zones, weather)
        T, w, rh, v = integrate_environment(
            T, w, N_mean, zones, selected, K, alpha, q_out,
            data["connections"], weather, cfg, dt_min, pressure
        )
        Tr = get_radiant_map(data["radiant"], timestamp, zones)
        for z in zone_ids:
            exp_t = experienced_temperature(T[z], rh[z], v[z], Tr[z], coef)
            pref, width = comfort_population(N_mean[z], cfg["comfort"], rng)
            cf = comfort_fraction(exp_t, pref, width)
            records.append({
                "scenario": scenario, "timestamp": timestamp, "zone_id": z,
                "air_temp_c": T[z], "humidity_ratio": w, "rh_pct": rh[z],
                "wind_speed_m_s": v[z], "radiant_temp_c": Tr[z],
                "experienced_temp_c": exp_t, "passengers": N_mean[z],
                "passengers_end": N_next[z], "exits": exits[z],
                "comfort_fraction": cf, "comfortable_persons": N_mean[z] * cf,
                "outdoor_exchange_m3_s": q_out[z],
            })
        N = N_next
    return pd.DataFrame(records), selected, K


def _annualized_capex(capex, lifetime, rate):
    if lifetime <= 0:
        raise ValueError("Device lifetime must be positive")
    if rate == 0:
        return capex / lifetime
    factor = rate * (1 + rate) ** lifetime / ((1 + rate) ** lifetime - 1)
    return capex * factor


def evaluate_scenarios(results, devices, cfg, dt_hours):
    econ = cfg["economics"]
    days = float(econ["operating_days_per_year"])
    daily_hours = results.timestamp.nunique() * dt_hours
    annual_hours = daily_hours * days
    base = results[results.scenario == "Baseline"].copy()
    base_key = base.set_index(["timestamp", "zone_id"])
    summaries = []
    zone_rows = []
    scenario_names = [s for s in results.scenario.unique() if s != "Baseline"]

    for scenario in scenario_names:
        cur = results[results.scenario == scenario].copy()
        cur_key = cur.set_index(["timestamp", "zone_id"])
        joined = cur_key.join(base_key[["comfort_fraction", "comfortable_persons"]], rsuffix="_baseline")
        joined["delta_comfortable_persons"] = joined["comfortable_persons"] - joined["comfortable_persons_baseline"]
        joined["delta_comfortable_person_hours"] = joined["delta_comfortable_persons"] * dt_hours
        daily_cph = float(joined.delta_comfortable_person_hours.sum())
        annual_cph = daily_cph * days
        # Also retain raw scenario comfort-hours for interpretation.
        annual_comfort_hours = float(cur.comfortable_persons.sum() * dt_hours * days)

        subset = _scenario_devices(devices, scenario)
        electricity_kwh = float((subset.power_kw.astype(float) * annual_hours).sum()) if len(subset) else 0.0
        water_m3 = float((pd.to_numeric(subset.get("water_l_per_h", pd.Series(0, index=subset.index)), errors="coerce").fillna(0).astype(float).sum() * annual_hours) / 1000.0)
        electricity_cost = electricity_kwh * float(econ["electricity_price_per_kwh"])
        water_cost = water_m3 * float(econ["water_price_per_m3"])
        capex_annual = maintenance = 0.0
        for _, d in subset.iterrows():
            capex_annual += _annualized_capex(float(d.get("capex", 0) or 0),
                                               float(d.get("lifetime_years", econ["default_lifetime_years"])),
                                               float(econ["discount_rate"]))
            maintenance += float(d.get("maintenance_per_year", 0) or 0)
        total_cost = electricity_cost + water_cost + capex_annual + maintenance
        benefit_value = annual_cph * float(econ["value_per_person_hour"])
        net_value = benefit_value - total_cost
        summaries.append({
            "Scenario": scenario,
            "Daily_Incremental_Comfortable_Person_Hours": daily_cph,
            "Annual_Incremental_Comfortable_Person_Hours": annual_cph,
            "Annual_Comfortable_Person_Hours": annual_comfort_hours,
            "Annual_Energy_kWh": electricity_kwh,
            "Annual_Water_m3": water_m3,
            "Annual_Electricity_Cost": electricity_cost,
            "Annual_Water_Cost": water_cost,
            "Annualized_Capital_Cost": capex_annual,
            "Annual_Maintenance_Cost": maintenance,
            "Annual_Total_Cost": total_cost,
            "Comfort_Benefit_Value": benefit_value,
            "Net_Annual_Value": net_value,
            "Comfort_Hours_per_kWh": annual_cph / electricity_kwh if electricity_kwh > 0 else np.nan,
            "Comfort_Hours_per_Dollar": annual_cph / total_cost if total_cost > 0 else np.nan,
        })
        for (timestamp, zone_id), row in joined.iterrows():
            zone_rows.append({"scenario": scenario, "timestamp": timestamp, "zone_id": zone_id,
                              "comfort_fraction_baseline": row.comfort_fraction_baseline,
                              "comfort_fraction_scenario": row.comfort_fraction,
                              "comfortable_persons_baseline": row.comfortable_persons_baseline,
                              "comfortable_persons_scenario": row.comfortable_persons,
                              "incremental_comfortable_persons": row.delta_comfortable_persons,
                              "incremental_comfortable_person_hours": row.delta_comfortable_person_hours})

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        metric_cols = ["Annual_Incremental_Comfortable_Person_Hours", "Comfort_Hours_per_kWh", "Comfort_Hours_per_Dollar"]
        norm = pd.DataFrame(index=summary.index)
        for col in metric_cols:
            values = summary[col].replace([np.inf, -np.inf], np.nan)
            low, high = values.min(skipna=True), values.max(skipna=True)
            if pd.isna(low) or pd.isna(high) or np.isclose(high, low):
                norm[col] = 1.0
            else:
                norm[col] = (values.fillna(low) - low) / (high - low)
        weights = econ["score_weights"]
        summary["Overall_Score"] = (float(weights["comfort_benefit"]) * norm[metric_cols[0]]
                                    + float(weights["energy_efficiency"]) * norm[metric_cols[1]]
                                    + float(weights["cost_efficiency"]) * norm[metric_cols[2]])
        summary = summary.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
    return summary, pd.DataFrame(zone_rows)


def make_evaluation_plots(summary, results_dir, figures_dir):
    if summary.empty:
        return
    cols = ["Annual_Incremental_Comfortable_Person_Hours", "Annual_Total_Cost", "Overall_Score"]
    labels = ["Annual incremental comfortable person-hours", "Annual total cost", "Overall score"]
    for col, label in zip(cols, labels):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(summary.Scenario, summary[col])
        ax.set_ylabel(label)
        ax.set_title(label + " by cooling scenario")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{col.lower()}_option2.png", dpi=160)
        plt.close(fig)


def run_full_evaluation(root: Path):
    cfg = load_config(root)
    data, _ = load_inputs(root, cfg)
    zones, station_cfg = generate_zone_geometry(data["zones"], cfg)
    data["zones"] = zones
    validate_inputs(data, float(cfg["geometry"].get("minimum_height_m", 8.0)))
    devices = assign_devices_to_zones(zones, data["devices"])
    warnings = validate_station_geometry(zones, station_cfg, devices, data.get("openings"))
    weather_df = data["weather"].sort_values("timestamp")
    timestamps = weather_df.timestamp.tolist()[:int(cfg["simulation"]["duration_intervals"])]
    if not timestamps:
        raise ValueError("No weather timestamps available")
    coef_path = root / cfg["files"]["regression_coefficients"]
    coef, coef_source = load_regression_coefficients(coef_path, cfg["model"]["regression_fallback"])
    print("Experienced-temperature coefficients:", coef_source)
    if "fallback" in coef_source.lower():
        print("WARNING: fallback coefficients are for pipeline testing only; benefit rankings are not scientifically valid until calibrated coefficients are supplied.")

    scenarios = ["Baseline", "Fan only", "Evaporative only", "Combined cooling"]
    frames, scenario_ks = [], {}
    seed = int(cfg["comfort"]["random_seed"])
    for scenario in scenarios:
        frame, selected, K = simulate_scenario(scenario, cfg, data, zones, devices, timestamps, coef, seed)
        frames.append(frame)
        scenario_ks[scenario] = K
        print(f"Completed scenario: {scenario} ({len(selected)} devices)")
    all_results = pd.concat(frames, ignore_index=True)
    dt_hours = float(cfg["simulation"]["interval_minutes"]) / 60.0
    summary, zone_comparison = evaluate_scenarios(all_results, devices, cfg, dt_hours)

    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    all_results.to_csv(results_dir / "scenario_simulation_results_option2.csv", index=False)
    summary.to_csv(results_dir / "overall_evaluation_option2.csv", index=False)
    zone_comparison.to_csv(results_dir / "zone_comfort_comparison_option2.csv", index=False)
    baseline = all_results[all_results.scenario == "Baseline"]
    baseline.to_csv(results_dir / "baseline_simulation_results_option2.csv", index=False)
    combined = all_results[all_results.scenario == "Combined cooling"]
    combined.to_csv(results_dir / "station_simulation_results_option2.csv", index=False)
    zones.to_csv(results_dir / "zone_geometry_generated_option2.csv", index=False)
    with (results_dir / "geometry_check_report_option2.txt").open("w", encoding="utf-8") as f:
        f.write("Geometry warnings (inputs were not auto-corrected):\n" if warnings else "No geometry warnings detected.\n")
        for item in warnings:
            f.write(f"- {item}\n")
    plot_station_geometry(zones, devices, data.get("openings"), figures_dir / "station_geometry_option2.png", station_cfg=station_cfg)
    make_evaluation_plots(summary, results_dir, figures_dir)
    # Restore detailed diagnostic plots from the earlier project version.
    # These time-series plots describe the Combined cooling scenario.
    make_plots(combined, scenario_ks["Combined cooling"], figures_dir)
    print("\nOverall evaluation (illustrative economics; check assumptions):")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Results written to: {results_dir}")
    return summary
