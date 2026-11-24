import numpy as np
import pandas as pd

def direction_factor(device_direction_deg, device_x, device_y, target_x, target_y):
    dx, dy = target_x - device_x, target_y - device_y
    if np.hypot(dx, dy) < 1e-9:
        return 1.0
    bearing = np.degrees(np.arctan2(dy, dx))
    diff = (bearing - device_direction_deg + 180) % 360 - 180
    theta = np.radians(diff)
    return float(np.cos(theta) ** 2) if abs(theta) <= 90 else 0.0

def spatial_influence(zones: pd.DataFrame, devices: pd.DataFrame):
    """Return K[device, zone], normalized allocation alpha[device, zone].
    Device locations remain fixed inputs. If z_m is supplied, vertical distance
    to the zone mid-height is included in the distance attenuation.
    """
    zone_ids = zones.zone_id.astype(str).tolist()
    device_ids = devices.device_id.astype(str).tolist()
    K = pd.DataFrame(0.0, index=device_ids, columns=zone_ids)
    alpha = K.copy()
    for _, d in devices.iterrows():
        for _, z in zones.iterrows():
            dx, dy = float(z.x_center_m)-float(d.x_m), float(z.y_center_m)-float(d.y_m)
            z_device = pd.to_numeric(pd.Series([d.get("z_m", np.nan)]), errors="coerce").iloc[0]
            dz = 0.0 if pd.isna(z_device) else float(z.height_m)/2.0 - float(z_device)
            r = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            g = direction_factor(d.direction_deg, d.x_m, d.y_m,
                                 z.x_center_m, z.y_center_m)
            k = np.exp(-r / max(float(d.decay_length_m), 1e-6)) * g
            K.loc[str(d.device_id), str(z.zone_id)] = k
        weights = K.loc[str(d.device_id)].values * zones.set_index("zone_id").loc[zone_ids, "area_m2"].values
        total = weights.sum()
        if total > 0:
            alpha.loc[str(d.device_id)] = weights / total
        else:
            # A device aimed away from all modeled zones has no modeled reach.
            # Keep a zero row rather than silently retaining unnormalized values.
            alpha.loc[str(d.device_id)] = 0.0
    if not K.index.is_unique or not K.columns.is_unique:
        raise ValueError("Spatial influence matrix requires unique device IDs and zone IDs.")
    return K, alpha

def outdoor_exchange(zones, weather_row):
    """Legacy aggregate opening model retained for compatibility.
    Geometry of openings is stored separately; exchange_coefficient remains
    the calibrated/assumed aggregate coefficient until a validated pressure-flow
    model is supplied.
    """
    out = {}
    for _, z in zones.iterrows():
        q_m3_h = float(z.exchange_coefficient) * float(z.opening_area_m2) * float(weather_row.outdoor_wind_m_s) * 3600
        out[str(z.zone_id)] = q_m3_h / 3600.0
    return out
