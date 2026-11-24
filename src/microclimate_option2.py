import numpy as np
import pandas as pd
from psychrometrics_option2 import humidity_ratio_from_rh, rh_from_humidity_ratio

def get_time_row(df, timestamp):
    exact = df[df.timestamp == timestamp]
    if not exact.empty:
        return exact.iloc[0]
    earlier = df[df.timestamp <= timestamp]
    return earlier.iloc[-1] if not earlier.empty else df.iloc[0]

def get_radiant_map(radiant_df, timestamp, zones):
    sub = radiant_df[radiant_df.timestamp == timestamp]
    if sub.empty:
        sub = radiant_df[radiant_df.timestamp <= timestamp].sort_values("timestamp").groupby("zone_id").tail(1)
    mapping = dict(zip(sub.zone_id.astype(str), sub.radiant_temp_c.astype(float)))
    return {str(z.zone_id): mapping.get(str(z.zone_id), float(z.initial_temp_c)) for _, z in zones.iterrows()}

def device_supply_conditions(device, weather, pressure):
    tout = float(weather.outdoor_temp_c)
    if pd.notna(device.get("supply_temp_c")):
        tsup = float(device.supply_temp_c)
    elif pd.notna(weather.get("wet_bulb_c")) and pd.notna(device.get("evaporative_efficiency")):
        tsup = tout - float(device.evaporative_efficiency) * (tout - float(weather.wet_bulb_c))
    else:
        tsup = tout
    if pd.notna(device.get("supply_humidity_ratio")):
        wsup = float(device.supply_humidity_ratio)
    else:
        wsup = humidity_ratio_from_rh(tout, float(weather.outdoor_rh_pct), pressure)
    return tsup, wsup

def integrate_environment(T, w, N_mean, zones, devices, K, alpha, q_out, connections,
                          weather, cfg, dt_minutes, pressure):
    """Explicit Euler substeps; temperatures are zone-specific, humidity ratio is station-wide."""
    model = cfg["model"]
    rho = float(cfg["simulation"]["air_density_kg_m3"])
    cp = float(cfg["simulation"]["cp_air_j_kg_k"])
    dt = dt_minutes * 60.0
    zone_ids = zones.zone_id.astype(str).tolist()
    volumes = {str(r.zone_id): float(r.area_m2) * float(r.height_m) for _, r in zones.iterrows()}
    device_data = {str(r.device_id): r for _, r in devices.iterrows()}
    radiant = {}
    # Airflow allocation by K * area; alpha is normalized over zones.
    q_device_zone = {}
    fan_speed = {z: 0.0 for z in zone_ids}
    evap_supply = {}
    for did, d in device_data.items():
        total_flow = float(d.airflow_m3_h) / 3600.0 * float(d.gamma)
        for z in zone_ids:
            q_device_zone[(did, z)] = total_flow * float(alpha.loc[did, z])
            if str(d.device_type).lower() == "fan":
                fan_speed[z] += float(d.fan_speed_gain_m_s or 0.0) * float(K.loc[did, z])
        if str(d.device_type).lower() in ["evaporative", "evaporative_cooler"]:
            evap_supply[did] = device_supply_conditions(d, weather, pressure)

    # Convert connection flow to symmetric mixing: one row defines an exchange pair.
    mix = []
    for _, r in connections.iterrows():
        mix.append((str(r.from_zone), str(r.to_zone), float(r.air_exchange_m3_h) / 3600.0))

    outdoor_w = humidity_ratio_from_rh(float(weather.outdoor_temp_c), float(weather.outdoor_rh_pct), pressure)
    total_volume = sum(volumes.values())
    qout_total = sum(q_out.values())

    steps = max(1, int(round((dt_minutes * 60) / max(1, int(cfg["simulation"]["integration_step_minutes"]) * 60))))
    h = dt / steps
    for _ in range(steps):
        dT = {z: 0.0 for z in zone_ids}
        # Outdoor ventilation.
        for z in zone_ids:
            dT[z] += q_out[z] / volumes[z] * (float(weather.outdoor_temp_c) - T[z])
        # Inter-zone air mixing; equal and opposite heat transfer.
        for a, b, q in mix:
            dT[a] += q / volumes[a] * (T[b] - T[a])
            dT[b] += q / volumes[b] * (T[a] - T[b])
        # Evaporative cooler supply-air sensible effect.
        for did, (tsup, wsup) in evap_supply.items():
            for z in zone_ids:
                q = q_device_zone[(did, z)]
                dT[z] += q / volumes[z] * (tsup - T[z])
        # Internal sensible heat: passenger load + approximate solar/surface gain.
        for _, zr in zones.iterrows():
            z = str(zr.zone_id)
            people = float(N_mean[z])
            q_people = people * 75.0  # W/person, illustrative sensible heat
            roof_cov = pd.to_numeric(pd.Series([zr.get("roof_coverage_fraction", np.nan)]), errors="coerce").iloc[0]
            roof_cov = 0.0 if pd.isna(roof_cov) else float(np.clip(roof_cov, 0.0, 1.0))
            non_roof_shade = float(np.clip(zr.get("shade_factor", 0.0), 0.0, 1.0))
            effective_shade = 1.0 - (1.0 - roof_cov) * (1.0 - non_roof_shade)
            solar_gain = float(weather.get("solar_w_m2", 0.0)) * float(zr.area_m2) * (1.0 - effective_shade) * 0.03
            q_surface = solar_gain
            dT[z] += (q_people + q_surface) / (rho * cp * volumes[z])
        for z in zone_ids:
            T[z] += h * dT[z]
        # Station-wide moisture balance.
        moisture_rate = qout_total / total_volume * (outdoor_w - w)
        for did, (tsup, wsup) in evap_supply.items():
            qsum = sum(q_device_zone[(did, z)] for z in zone_ids)
            moisture_rate += qsum / total_volume * (wsup - w)
        # Illustrative passenger moisture source, kg water/s per person.
        people_total = sum(float(N_mean[z]) for z in zone_ids)
        vapor_kg_s = people_total * 0.000015
        rho_d = float(cfg["simulation"]["dry_air_density_kg_m3"])
        moisture_rate += vapor_kg_s / max(rho_d * total_volume, 1e-9)
        w = max(0.0001, w + h * moisture_rate)

    rh = {z: rh_from_humidity_ratio(T[z], w, pressure) for z in zone_ids}
    v = {}
    for _, zr in zones.iterrows():
        z = str(zr.zone_id)
        v[z] = float(zr.get("background_wind_m_s", model["default_background_wind_m_s"])) + min(
            float(model["max_fan_added_speed_m_s"]), fan_speed[z]
        )
    return T, w, rh, v
