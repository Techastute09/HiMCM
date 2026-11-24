from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def make_plots(results, spatial_K, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return
    for metric, ylabel, filename in [
        ("air_temp_c", "Air temperature (°C)", "temperature_by_zone_option2.png"),
        ("rh_pct", "Relative humidity (%)", "humidity_by_zone_option2.png"),
        ("passengers", "Passengers", "passengers_by_zone_option2.png"),
        ("experienced_temp_c", "Experienced temperature (°C)", "experienced_temperature_option2.png"),
        ("comfort_fraction", "Comfort fraction", "comfort_fraction_option2.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for zone, g in results.groupby("zone_id"):
            ax.plot(pd.to_datetime(g.timestamp), g[metric], marker="o", label=zone)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(spatial_K.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(spatial_K.columns)), spatial_K.columns)
    ax.set_yticks(range(len(spatial_K.index)), spatial_K.index)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Device")
    ax.set_title("Spatial influence K(device, zone)")
    fig.colorbar(im, ax=ax, label="Relative influence")
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_influence_option2.png", dpi=160)
    plt.close(fig)
