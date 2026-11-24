import math

def saturation_vapor_pressure_pa(temp_c):
    # Buck-type approximation over ordinary outdoor temperature ranges.
    return 611.21 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))

def humidity_ratio_from_rh(temp_c, rh_pct, pressure_pa=101325.0):
    pws = saturation_vapor_pressure_pa(float(temp_c))
    pv = max(0.0, min(1.0, float(rh_pct) / 100.0)) * pws
    return 0.62198 * pv / max(pressure_pa - pv, 1.0)

def rh_from_humidity_ratio(temp_c, w, pressure_pa=101325.0):
    pws = saturation_vapor_pressure_pa(float(temp_c))
    pv = pressure_pa * float(w) / (0.62198 + float(w))
    return max(0.0, min(100.0, 100.0 * pv / max(pws, 1.0)))
