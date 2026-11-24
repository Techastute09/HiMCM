from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all(df_flow, results, output_dir="figures"):
    """
    Generate four publication-style figures:
      1. Passenger flow over time
      2. Annual energy and annual cost comparison
      3. Comfort benefit versus annual cost
      4. Sensitivity heatmap for arrival multiplier and coverage multiplier
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: passenger flow
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df_flow))
    labels = df_flow["time"].astype(str) if "time" in df_flow else x
    ax.plot(x, df_flow["passengers_waiting"], marker="o", label="Waiting passengers")
    ax.bar(x, df_flow["arrivals_simulated"], alpha=0.25, label="Arrivals in interval")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of passengers")
    ax.set_title("Bus Stop Passenger Flow")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure1_passenger_flow_option2.png", dpi=220)
    plt.close(fig)

    # Figure 2: annual energy and annual cost (separate axes)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    names = results["Device"]
    x = np.arange(len(names))
    width = 0.36
    ax1.bar(x - width / 2, results["annual_energy_kwh"], width,
            label="Annual energy (kWh)")
    ax1.set_ylabel("Annual energy (kWh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, results["annual_total_cost"], width,
            alpha=0.65, label="Annual cost ($)")
    ax2.set_ylabel("Annualized cost ($)")
    ax1.set_title("Annual Energy and Cost by Cooling Device")
    ax1.grid(axis="y", alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "figure2_energy_cost_option2.png", dpi=220)
    plt.close(fig)

    # Figure 3: benefit versus cost
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(results["annual_total_cost"],
               results["Annual_Comfort_Benefit"], s=90)
    for _, row in results.iterrows():
        ax.annotate(row["Device"],
                    (row["annual_total_cost"], row["Annual_Comfort_Benefit"]),
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Annualized total cost ($)")
    ax.set_ylabel("Annual comfort benefit (person-hour × degree-equivalent)")
    ax.set_title("Comfort Benefit versus Annual Cost")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure3_benefit_vs_cost_option2.png", dpi=220)
    plt.close(fig)

    # Figure 4: sensitivity to arrival volume and coverage.
    # This is a transparent proportional sensitivity approximation:
    # comfort benefit scales with both passenger volume and coverage.
    arrival_multipliers = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    coverage_multipliers = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    baseline = results.set_index("Device")["Annual_Comfort_Benefit"]
    baseline_cost = results.set_index("Device")["annual_total_cost"]

    # Compare the device with the highest baseline overall score.
    best = results.iloc[0]["Device"]
    benefit = float(baseline.loc[best])
    cost = float(baseline_cost.loc[best])
    grid = np.zeros((len(coverage_multipliers), len(arrival_multipliers)))

    # Heatmap value is scaled comfort benefit per annual dollar.
    for iy, cm in enumerate(coverage_multipliers):
        for ix, am in enumerate(arrival_multipliers):
            grid[iy, ix] = benefit * am * cm / cost if cost > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(arrival_multipliers)))
    ax.set_xticklabels([f"{v:.2f}×" for v in arrival_multipliers])
    ax.set_yticks(np.arange(len(coverage_multipliers)))
    ax.set_yticklabels([f"{v:.2f}×" for v in coverage_multipliers])
    ax.set_xlabel("Passenger arrival multiplier")
    ax.set_ylabel("Coverage multiplier")
    ax.set_title(f"Sensitivity: {best} benefit per dollar")
    fig.colorbar(im, ax=ax, label="Scaled benefit per dollar")
    fig.tight_layout()
    fig.savefig(out / "figure4_sensitivity_heatmap_option2.png", dpi=220)
    plt.close(fig)
