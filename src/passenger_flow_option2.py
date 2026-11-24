import numpy as np

def advance_passengers(N, zones, arrivals_row, transitions, dt_hours):
    """
    Deterministic expected-count network model with competing movement hazards.
    For each origin zone, outgoing transitions and system exits share one total
    hazard, so the sum of departures cannot exceed the available population.
    Counts are expected values and may therefore be fractional.
    """
    ids = zones.zone_id.astype(str).tolist()
    n0 = {z: float(N.get(z, 0.0)) + float(arrivals_row.get(z, 0.0)) for z in ids}
    incoming = {z: 0.0 for z in ids}
    exits = {z: 0.0 for z in ids}
    outgoing = {z: [] for z in ids}

    for _, row in transitions.iterrows():
        origin, destination = str(row.from_zone), str(row.to_zone)
        rate = max(0.0, float(row.transition_rate_per_hour))
        outgoing[origin].append((destination, rate))

    exit_rates = transitions.groupby("from_zone").system_exit_rate_per_hour.max().to_dict()
    for origin in ids:
        exit_rate = max(0.0, float(exit_rates.get(origin, 0.0)))
        transition_rate_sum = sum(rate for _, rate in outgoing[origin])
        total_hazard = transition_rate_sum + exit_rate
        if total_hazard <= 0:
            continue
        total_departure = n0[origin] * (1.0 - np.exp(-total_hazard * dt_hours))
        for destination, rate in outgoing[origin]:
            incoming[destination] += total_departure * rate / total_hazard
        exits[origin] = total_departure * exit_rate / total_hazard

    n1 = {}
    mean_n = {}
    for z in ids:
        moved_out = sum(n0[z] * (1.0 - np.exp(-((sum(r for _, r in outgoing[z]) + max(0.0, float(exit_rates.get(z, 0.0)))) * dt_hours)))
                        * r / max(sum(r for _, r in outgoing[z]) + max(0.0, float(exit_rates.get(z, 0.0))), 1e-12)
                        for _, r in outgoing[z])
        n1[z] = max(0.0, n0[z] - moved_out - exits[z] + incoming[z])
        mean_n[z] = 0.5 * (n0[z] + n1[z])
    return n1, mean_n, exits
