from pathlib import Path
import numpy as np
import pandas as pd

# Bus Stop regression model: full second-order polynomial in Ta, Tr, RH, and v.
# CSV terms are mapped to these canonical coefficient names.
TERM_ALIASES = {
    "intercept": {"intercept", "const", "constant", "bias"},
    "ta": {"ta", "air_temp", "air_temperature", "temperature"},
    "tr": {"tr", "radiant_temp", "radiant_temperature", "mean_radiant_temperature"},
    "rh": {"rh", "relative_humidity", "rh_pct"},
    "vel": {"vel", "v", "wind_speed", "air_speed", "velocity"},
    "ta2": {"ta^2", "ta**2", "ta2"},
    "ta_tr": {"ta tr", "ta*tr", "ta_tr"},
    "ta_rh": {"ta rh", "ta*rh", "ta_rh"},
    "ta_vel": {"ta vel", "ta*vel", "ta_vel", "ta v", "ta*v"},
    "tr2": {"tr^2", "tr**2", "tr2"},
    "tr_rh": {"tr rh", "tr*rh", "tr_rh"},
    "tr_vel": {"tr vel", "tr*vel", "tr_vel", "tr v", "tr*v"},
    "rh2": {"rh^2", "rh**2", "rh2"},
    "rh_vel": {"rh vel", "rh*vel", "rh_vel", "rh v", "rh*v"},
    "vel2": {"vel^2", "vel**2", "v^2", "v**2", "vel2"},
}

REQUIRED_POLYNOMIAL_TERMS = tuple(TERM_ALIASES.keys())


def _normalize_term(term):
    return " ".join(str(term).strip().lower().replace("×", "*").split())


def load_regression_coefficients(path, fallback=None):
    """
    Load the Bus Stop second-order polynomial regression coefficients.

    Expected CSV format: two columns, `term` and `coefficient`.
    All 15 terms (intercept + 4 linear + 10 quadratic/interaction terms)
    must be present. A partial coefficient file is rejected to prevent
    silently evaluating a different regression equation.
    """
    p = Path(path)
    if not p.exists():
        if fallback is None:
            raise FileNotFoundError(f"Regression coefficient file not found: {p}")
        # Preserve the legacy fallback only for pipeline smoke tests.
        return dict(fallback), "fallback coefficients (NOT the Bus Stop polynomial)"

    df = pd.read_csv(p)
    if not {"term", "coefficient"}.issubset(df.columns):
        raise ValueError(
            "Regression coefficient CSV must contain 'term' and 'coefficient' columns."
        )

    alias_lookup = {}
    for canonical, aliases in TERM_ALIASES.items():
        for alias in aliases:
            alias_lookup[_normalize_term(alias)] = canonical

    coef = {}
    for _, row in df.iterrows():
        term = _normalize_term(row["term"])
        if term in alias_lookup:
            canonical = alias_lookup[term]
            coef[canonical] = float(row["coefficient"])

    missing = [term for term in REQUIRED_POLYNOMIAL_TERMS if term not in coef]
    if missing:
        raise ValueError(
            "Incomplete Bus Stop polynomial regression coefficients. "
            f"Missing terms: {missing}"
        )

    return coef, f"loaded full second-order Bus Stop regression from {p}"


def experienced_temperature(temp_c, rh_pct, wind_m_s, radiant_c, coef):
    """
    Calculate experienced temperature using the exact second-order
    Bus Stop regression equation.

    Inputs:
      temp_c: air temperature Ta (°C)
      rh_pct: relative humidity RH (%)
      wind_m_s: air velocity v (m/s)
      radiant_c: radiant temperature Tr (°C)
    """
    ta = float(temp_c)
    tr = float(radiant_c)
    rh = float(rh_pct)
    vel = float(wind_m_s)

    return float(
        coef["intercept"]
        + coef["ta"] * ta
        + coef["tr"] * tr
        + coef["rh"] * rh
        + coef["vel"] * vel
        + coef["ta2"] * ta**2
        + coef["ta_tr"] * ta * tr
        + coef["ta_rh"] * ta * rh
        + coef["ta_vel"] * ta * vel
        + coef["tr2"] * tr**2
        + coef["tr_rh"] * tr * rh
        + coef["tr_vel"] * tr * vel
        + coef["rh2"] * rh**2
        + coef["rh_vel"] * rh * vel
        + coef["vel2"] * vel**2
    )


def comfort_population(zone_people, cfg, rng):
    n = max(0, int(round(zone_people)))
    mean_t = float(cfg["preferred_temp_mean_c"])
    sd_t = float(cfg["preferred_temp_sd_c"])
    mean_w = float(cfg["comfort_width_mean_c"])
    sd_w = float(cfg["comfort_width_sd_c"])
    preferred = rng.normal(mean_t, sd_t, n)
    width = np.maximum(0.2, rng.normal(mean_w, sd_w, n))
    return preferred, width


def comfort_fraction(experienced_temp, preferred, width):
    if len(preferred) == 0:
        return 0.0
    return float(np.mean(np.abs(experienced_temp - preferred) <= width / 2.0))
