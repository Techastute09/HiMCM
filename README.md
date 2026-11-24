# HiMCM Cooling System Evaluation and Optimization

This is a runnable Python starter project for connecting:
1. Passenger-flow simulation
2. Individual comfort-discomfort improvement
3. Annualized cooling equipment costs
4. Overall comparison and four figures

## Important data note

`data/bus_stop_environment_option2.csv` contains **illustrative synthetic values only**.
It is provided to test the code pipeline. Replace it with your actual bus-stop observations
and the before/after temperatures predicted by your calibrated regression model.

The default comfort-distribution parameters, device coverage fractions, device power,
capital costs, water use, maintenance, electricity price, and water price are also
illustrative assumptions. They must be sourced, calibrated, or tested with sensitivity
analysis before being used as final HiMCM results.

## Files

- `main_option2.py`: run the full workflow
- `src/pedestrian_flow_option2.py`: Poisson arrival and waiting-population model
- `src/comfort_benefit_option2.py`: Monte Carlo comfort population and discomfort reduction
- `src/cooling_cost_option2.py`: annualized capital and operating costs
- `src/overall_evaluation_option2.py`: integrates benefits and costs
- `src/plot_results_option2.py`: generates four figures
- `data/bus_stop_environment_option2.csv`: synthetic example input
- `figures/`: generated plots
- `results/`: generated CSV outputs

## Install

```bash
python -m pip install -r requirements_option2.txt
```

## Run

From this project directory:

```bash
python main_option2.py
```

The program writes result CSVs to `results/` and four PNG figures to `figures/`.

## Input CSV columns

| Column | Meaning |
|---|---|
| `time` | time label for the interval |
| `arrival_rate` | expected passenger arrivals per hour |
| `boarding_capacity` | number of passengers who can board during this interval |
| `T_before` | baseline experienced/apparent temperature from your calibrated model |
| `T_fan` | experienced/apparent temperature under fan scenario |
| `T_evap` | experienced/apparent temperature under evaporative cooler scenario |
| `T_large` | experienced/apparent temperature under large cooler scenario |

Each row represents 15 minutes in the sample data. Change `dt_hours` in `main_option2.py`
if your interval differs.

## Modeling cautions

- `boarding_capacity` is treated as the number of passengers actually boarding in that
  interval. If you have actual boarding counts, use those rather than a capacity estimate.
- The Poisson arrival model assumes arrivals within an interval follow a Poisson process.
  Validate this assumption against observed counts; peak-period arrivals may be clustered.
- The annualization assumes the sample day represents a typical operating day and is
  repeated 300 days per year. Replace this with a representative annual schedule or
  seasonal scenarios.
- The comfort score is a constructed discomfort-distance metric, not an official PMV,
  SET, UTCI, or TSV metric.
- The sample code applies the same generated comfort population to each time interval.
  For a more rigorous model, use your fitted individual comfort distribution and
  consider uncertainty in the regression predictions.
- The sensitivity heatmap is a proportional scenario analysis for passenger volume and
  coverage, not a new physical simulation.

## Units

- Temperatures: °C
- Power: kW
- Energy: kWh/year
- Water: m³/year
- Costs: USD/year
- Comfort benefit: discomfort-degree-equivalent × passenger-hour
