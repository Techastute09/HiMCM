import numpy as np
import pandas as pd


def simulate_pedestrian_flow(
    df: pd.DataFrame,
    dt_hours: float = 0.25,
    initial_people: int = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate passenger arrivals and waiting population.

    Required columns:
      arrival_rate: expected arrivals per hour
      boarding_capacity: maximum passengers boarding during this interval

    If actual observed arrivals/boardings are available, use those instead
    of simulated arrivals and treat this function as an optional scenario model.
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    required = {"arrival_rate", "boarding_capacity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rng = np.random.default_rng(random_seed)
    out = df.copy()
    waiting = int(initial_people)
    arrivals_list, boarded_list, waiting_list = [], [], []

    for row in out.itertuples(index=False):
        arrival_rate = max(0.0, float(getattr(row, "arrival_rate")))
        capacity = max(0, int(getattr(row, "boarding_capacity")))
        arrivals = int(rng.poisson(arrival_rate * dt_hours))
        available = waiting + arrivals
        boarded = min(available, capacity)
        waiting = available - boarded

        arrivals_list.append(arrivals)
        boarded_list.append(boarded)
        waiting_list.append(waiting)

    out["arrivals_simulated"] = arrivals_list
    out["boarded_simulated"] = boarded_list
    out["passengers_waiting"] = waiting_list
    return out
