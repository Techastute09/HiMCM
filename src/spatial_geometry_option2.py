import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def generate_zone_geometry(zones, cfg):
    """Use explicit geometry from CSV. Never auto-resize or auto-reposition zones."""
    zones = zones.copy()
    geom = cfg.get("geometry", {})
    h_min = float(geom.get("minimum_height_m", 8.0))
    required = ["area_m2", "length_m", "width_m", "height_m", "x_center_m", "y_center_m"]
    missing = [c for c in required if c not in zones.columns]
    if missing:
        raise ValueError(f"station_zones_option2.csv missing explicit geometry columns: {missing}")
    for c in required:
        zones[c] = pd.to_numeric(zones[c], errors="coerce")
    if zones[required].isna().any().any():
        raise ValueError("Explicit zone dimensions, positions, area and height must be numeric.")
    if (zones[["area_m2", "length_m", "width_m", "height_m"]] <= 0).any().any():
        raise ValueError("Zone area, length, width and height must be positive.")
    if (zones.height_m < h_min).any():
        bad = zones.loc[zones.height_m < h_min, "zone_id"].astype(str).tolist()
        raise ValueError(f"Zone heights must be >= {h_min:g} m. Check: {bad}")
    # Keep the supplied center coordinates and dimensions exactly as entered.
    station_cfg = cfg.setdefault("station_geometry", {})
    if station_cfg.get("length_m") is None or station_cfg.get("width_m") is None:
        raise ValueError("Set station_geometry.length_m and width_m explicitly in station_config_option2.yaml.")
    return zones, station_cfg

def station_bounds(station_cfg):
    if not station_cfg or station_cfg.get("shape", "rectangle") != "rectangle": return None
    L, W = float(station_cfg["length_m"]), float(station_cfg["width_m"])
    if L <= 0 or W <= 0: raise ValueError("Station length and width must be positive.")
    return float(station_cfg.get("origin_x_m",0)), float(station_cfg.get("origin_y_m",0)), L, W

def assign_devices_to_zones(zones, devices):
    """Assign annotation labels only; fixed device positions are not changed."""
    out=devices.copy()
    if "zone_id" not in out: out["zone_id"]=""
    out["zone_id"]=out.zone_id.fillna("").astype(str).astype("object")
    for idx,d in out.iterrows():
        if str(d.zone_id).strip() and str(d.zone_id).lower()!="nan": continue
        dx=zones.x_center_m.to_numpy(float)-float(d.x_m)
        dy=zones.y_center_m.to_numpy(float)-float(d.y_m)
        out.at[idx,"zone_id"]=str(zones.iloc[int(np.argmin(np.hypot(dx,dy)))].zone_id)
    if "z_m" not in out: out["z_m"]=np.nan
    return out

def geometry_diagnostics(zones, station_cfg, devices=None, openings=None):
    """Return warnings; geometry inconsistencies do not silently rewrite inputs."""
    warnings=[]
    sb=station_bounds(station_cfg)
    rects={}
    for _,z in zones.iterrows():
        x0=float(z.x_center_m)-float(z.length_m)/2
        y0=float(z.y_center_m)-float(z.width_m)/2
        x1=x0+float(z.length_m); y1=y0+float(z.width_m)
        rects[str(z.zone_id)]=(x0,y0,x1,y1)
        if abs(float(z.length_m)*float(z.width_m)-float(z.area_m2))/float(z.area_m2)>0.05:
            warnings.append(f"{z.zone_id}: L×W differs from area_m2 by more than 5%.")
        if sb:
            sx,sy,sl,sw=sb
            if x0<sx or y0<sy or x1>sx+sl or y1>sy+sw:
                warnings.append(f"{z.zone_id}: rectangle extends outside station boundary.")
    ids=list(rects)
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a,b=rects[ids[i]],rects[ids[j]]
            overlap_x=min(a[2],b[2])-max(a[0],b[0])
            overlap_y=min(a[3],b[3])-max(a[1],b[1])
            if overlap_x>1e-6 and overlap_y>1e-6:
                warnings.append(f"{ids[i]} and {ids[j]} overlap by approx {overlap_x*overlap_y:.2f} m².")
    if devices is not None and sb:
        sx,sy,sl,sw=sb
        for _,d in devices.iterrows():
            if not (sx<=float(d.x_m)<=sx+sl and sy<=float(d.y_m)<=sy+sw):
                warnings.append(f"Fixed device {d.device_id} lies outside station boundary.")
    if openings is not None and not openings.empty and sb:
        sx,sy,sl,sw=sb
        for _,o in openings.iterrows():
            if not (sx<=float(o.x_m)<=sx+sl and sy<=float(o.y_m)<=sy+sw):
                warnings.append(f"Opening {o.opening_id} lies outside station boundary.")
    return warnings

def validate_station_geometry(zones, station_cfg, devices=None, openings=None):
    # Intentionally non-blocking: report geometry concerns in main workflow.
    return geometry_diagnostics(zones, station_cfg, devices, openings)

def plot_station_geometry(zones, devices, openings, output_path, station_cfg=None):
    fig,ax=plt.subplots(figsize=(13,8))
    sb=station_bounds(station_cfg or {})
    if sb:
        sx,sy,sl,sw=sb
        ax.add_patch(Rectangle((sx,sy),sl,sw,fill=False,edgecolor="black",linewidth=2.8))
    for _,z in zones.iterrows():
        L,W=float(z.length_m),float(z.width_m)
        x0,y0=float(z.x_center_m)-L/2,float(z.y_center_m)-W/2
        roof=float(z.roof_coverage_fraction) if pd.notna(z.get("roof_coverage_fraction")) else 0.0
        ax.add_patch(Rectangle((x0,y0),L,W,fill=False,edgecolor="0.2",linewidth=1.6,zorder=2))
        ax.add_patch(Rectangle((x0,y0),L,W,facecolor="0.75",alpha=0.18+0.35*roof,
                               hatch="///" if roof>0 else None,edgecolor="0.5",zorder=1.5))
        ax.text(float(z.x_center_m),float(z.y_center_m),
                f"{z.zone_id}\\nA={float(z.area_m2):g} m²\\n{L:.2f} × {W:.2f} m\\nH={float(z.height_m):.1f} m\\nRoof={roof:.0%}",
                ha="center",va="center",fontsize=8.5,zorder=4)
    if devices is not None:
        for _,d in devices.iterrows():
            ax.scatter(float(d.x_m),float(d.y_m),marker="*",s=220,edgecolor="black",zorder=6)
            ax.annotate(str(d.device_id),(float(d.x_m),float(d.y_m)),xytext=(6,6),textcoords="offset points",fontsize=9)
            ang=np.radians(float(d.direction_deg))
            ax.arrow(float(d.x_m),float(d.y_m),2*np.cos(ang),2*np.sin(ang),head_width=.45,
                     length_includes_head=True,color="tab:orange",zorder=6)
    if openings is not None and not openings.empty:
        for _,o in openings.iterrows():
            ax.scatter(float(o.x_m),float(o.y_m),marker="s",s=90,color="tab:cyan",edgecolor="black",zorder=7)
            ax.annotate(f"{o.opening_id}\\n{float(o.width_m):g}×{float(o.height_m):g} m",
                        (float(o.x_m),float(o.y_m)),xytext=(5,-10),textcoords="offset points",fontsize=8)
    ax.set_title("Semi-open Station — Explicit Zone Geometry (Option 2)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True,linestyle="--",alpha=.35); ax.set_aspect("equal",adjustable="box")
    if sb:
        sx,sy,sl,sw=sb; ax.set_xlim(sx-1,sx+sl+1); ax.set_ylim(sy-1,sy+sw+1)
    ax.text(.01,-.12,"Equal X/Y scale. Dimensions and positions are explicit input assumptions; check overlap and boundary warnings.",
            transform=ax.transAxes,fontsize=8.5)
    fig.tight_layout(); fig.savefig(output_path,dpi=180,bbox_inches="tight"); plt.close(fig)
