import numpy as np


def generate_population(
    n=10000,
    #this means the average preferred temperature is 23.2925 degrees Celsius from dataset training
    mean_preferred_temp=23.2925,
    std_preferred_temp=3.0,
    mean_log_width=np.log(6.0),
    std_log_width=0.30,
    correlation=0.20,
    seed=42,
):
    """
    Monte Carlo population of individual comfort preferences.

    Parameters are illustrative defaults. Replace with estimates from
    the selected comfort survey/dataset and your calibration workflow.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not -1 <= correlation <= 1:
        raise ValueError("correlation must be between -1 and 1")

    cov = np.array([
        [std_preferred_temp**2,
         correlation * std_preferred_temp * std_log_width],
        [correlation * std_preferred_temp * std_log_width,
         std_log_width**2],
    ])
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(
        [mean_preferred_temp, mean_log_width], cov, size=n
    )
    preferred = z[:, 0]
    width = np.exp(z[:, 1])
    return preferred, width


def discomfort(temp, preferred, width):
    """Distance outside an individual's comfort interval, in °C."""
    return np.maximum(0.0, np.abs(temp - preferred) - width / 2.0)


def expected_benefit(temp_before, temp_after, preferred, width):
    """
    Mean reduction in discomfort across the simulated population.
    Positive = improvement; negative = worsening.
    """
    before = discomfort(temp_before, preferred, width)
    after = discomfort(temp_after, preferred, width)
    return float(np.mean(before - after))
