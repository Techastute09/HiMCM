from pathlib import Path
import yaml
import pandas as pd

def load_config(project_root):
    with (project_root / "data/station_config_option2.yaml").open("r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_inputs(project_root,cfg):
    paths={k:project_root/v for k,v in cfg["files"].items()}
    required=["zones","devices","connections","weather","arrivals","transitions","radiant"]
    missing=[str(paths[k]) for k in required if not paths[k].exists()]
    if missing: raise FileNotFoundError("Missing input files:\\n"+"\\n".join(missing))
    data={k:pd.read_csv(paths[k]) for k in required}
    opath=paths.get("openings")
    cols=["opening_id","zone_id","wall_side","x_m","y_m","z_m","width_m","height_m","opening_type","orientation_deg"]
    data["openings"]=pd.read_csv(opath) if opath and opath.exists() else pd.DataFrame(columns=cols)
    for k in ["weather","arrivals","radiant"]:
        if "timestamp" in data[k]: data[k]["timestamp"]=pd.to_datetime(data[k]["timestamp"])
    return data,paths

def validate_inputs(data,minimum_height_m=8.0):
    z=data["zones"]
    if z.empty or "zone_id" not in z: raise ValueError("At least one zone_id is required.")
    if z.zone_id.duplicated().any(): raise ValueError("zone_id values must be unique.")
    for c in ["area_m2","height_m","x_center_m","y_center_m","length_m","width_m"]:
        if c not in z: raise ValueError(f"Missing zone column: {c}")
    if (pd.to_numeric(z.area_m2,errors="coerce")<=0).any(): raise ValueError("Zone areas must be positive.")
    if (pd.to_numeric(z.height_m,errors="coerce")<minimum_height_m).any(): raise ValueError(f"All zone heights must be at least {minimum_height_m:g} m.")
    if (z[["length_m","width_m"]].apply(pd.to_numeric,errors="coerce")<=0).any().any(): raise ValueError("Generated zone dimensions must be positive.")
    if (((z.length_m*z.width_m-z.area_m2).abs()/z.area_m2)>0.05).any(): raise ValueError("Zone length × width must agree with area within 5%.")
    required={"connections":["from_zone","to_zone","air_exchange_m3_h"],
      "transitions":["from_zone","to_zone","transition_rate_per_hour","system_exit_rate_per_hour"],
      "arrivals":["timestamp","zone_id","arrivals"],
      "weather":["timestamp","outdoor_temp_c","outdoor_rh_pct","outdoor_wind_m_s"],
      "radiant":["timestamp","zone_id","radiant_temp_c"],
      "devices":["device_id","device_type","x_m","y_m","direction_deg","airflow_m3_h","decay_length_m","gamma","power_kw"]}
    for k,cols in required.items():
        miss=set(cols)-set(data[k].columns)
        if miss: raise ValueError(f"{k} missing columns: {sorted(miss)}")
    devices=data["devices"]
    if devices.device_id.astype(str).duplicated().any():
        dup=devices.loc[devices.device_id.astype(str).duplicated(keep=False),"device_id"].astype(str).unique().tolist()
        raise ValueError(f"device_id values must be unique; duplicates: {dup}")
    if devices.empty:
        raise ValueError("At least one device row is required in the input CSV; use scenario filtering to create the no-device baseline.")
    for col in ["airflow_m3_h","decay_length_m","gamma","power_kw"]:
        vals=pd.to_numeric(devices[col],errors="coerce")
        if vals.isna().any(): raise ValueError(f"devices.{col} must contain numeric values.")
    if (pd.to_numeric(devices.airflow_m3_h)<=0).any(): raise ValueError("Device airflow_m3_h must be positive.")
    if (pd.to_numeric(devices.decay_length_m)<=0).any(): raise ValueError("Device decay_length_m must be positive.")
    if ((pd.to_numeric(devices.gamma)<0)|(pd.to_numeric(devices.gamma)>1)).any(): raise ValueError("Device gamma must be between 0 and 1.")
    ids=set(z.zone_id.astype(str))
    for k in ["connections","transitions","arrivals","radiant"]:
        d=data[k]
        refs=set(d.zone_id.astype(str)) if "zone_id" in d else set(d.from_zone.astype(str))|set(d.to_zone.astype(str))
        if refs-ids: raise ValueError(f"{k} refers to unknown zones: {sorted(refs-ids)}")
    op=data["openings"]
    if not op.empty:
        need={"opening_id","zone_id","x_m","y_m","width_m","height_m"}
        if need-set(op.columns): raise ValueError(f"openings missing columns: {sorted(need-set(op.columns))}")
        if set(op.zone_id.astype(str))-ids: raise ValueError("Opening references unknown zone.")
        if (op[["width_m","height_m"]]<=0).any().any(): raise ValueError("Opening dimensions must be positive.")
    if (data["connections"].air_exchange_m3_h<0).any(): raise ValueError("Air exchange cannot be negative.")
    if (data["transitions"].transition_rate_per_hour<0).any(): raise ValueError("Passenger transition rates cannot be negative.")
