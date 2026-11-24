def capital_recovery_factor(discount_rate: float, lifetime_years: int) -> float:
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    if discount_rate < 0:
        raise ValueError("discount_rate cannot be negative")
    if discount_rate == 0:
        return 1.0 / lifetime_years
    r = discount_rate
    n = lifetime_years
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def calculate_annual_cost(
    power_kw: float,
    operating_hours_per_year: float,
    electricity_price_per_kwh: float,
    capex: float,
    lifetime_years: int,
    discount_rate: float,
    water_m3_per_year: float = 0.0,
    water_price_per_m3: float = 0.0,
    maintenance_per_year: float = 0.0,
):
    """Return annual energy and annualized cost components."""
    values = [
        power_kw, operating_hours_per_year, electricity_price_per_kwh,
        capex, water_m3_per_year, water_price_per_m3, maintenance_per_year
    ]
    if any(v < 0 for v in values):
        raise ValueError("Cost and operating parameters cannot be negative")

    energy = power_kw * operating_hours_per_year
    electricity = energy * electricity_price_per_kwh
    water = water_m3_per_year * water_price_per_m3
    annual_capital = capex * capital_recovery_factor(
        discount_rate, lifetime_years
    )
    total = annual_capital + electricity + water + maintenance_per_year

    return {
        "annual_energy_kwh": energy,
        "annual_electricity_cost": electricity,
        "annual_water_cost": water,
        "annualized_capital_cost": annual_capital,
        "annual_maintenance_cost": maintenance_per_year,
        "annual_total_cost": total,
    }
