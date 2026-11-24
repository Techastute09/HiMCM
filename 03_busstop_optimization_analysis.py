from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pedestrian_flow import simulate_pedestrian_flow
from overall_evaluation import evaluate_devices
from plot_results import plot_all
from calculate_experienced_t import (
    predict_experienced_temperature,
    calculate_device_temperatures
)

def main():
    data_path = ROOT / "data" / "bus_stop_environment.csv"
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    # Set dt_hours to match the interval represented by each row.
    dt_hours = 0.25

    flow = simulate_pedestrian_flow(
        df,
        dt_hours=dt_hours,
        initial_people=0,
        random_seed=42,
    )
    flow.to_csv(results_dir / "pedestrian_flow_results.csv", index=False)

    results = evaluate_devices(
        flow,
        dt_hours=dt_hours,
        operating_days_per_year=300,
        electricity_price=0.25,  # $/kWh: replace with local tariff
        water_price=2.00,        # $/m3: replace with local tariff
        lifetime_years=10,
        discount_rate=0.05,
    )
    results.to_csv(results_dir / "overall_comparison.csv", index=False)

    plot_all(flow, results, output_dir=figures_dir)

    print("\\n=== Overall Device Comparison ===")
    print(results.to_string(index=False))
    print(f"\\nBest device under current assumptions: {results.iloc[0]['Device']}")
    print(f"Results saved in: {results_dir}")
    print(f"Figures saved in: {figures_dir}")

    # Env4, just for testing the experienced temperature calculation
    env4 = {
        "ta_h": 35,
        "vel_h": 0.5,
        "rh": 45,
        "tr": 38
    }

    temperature_results = calculate_device_temperatures(
        env4=env4
    )

    for device, temperature in temperature_results.items():
        print(f"{device}: {temperature:.3f}")

if __name__ == "__main__":
    main()
