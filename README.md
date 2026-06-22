# SDG Shopfloor Capacity Simulation

Discrete-event simulation (DES) of the SDG shopfloor for the 1CM140 Smart Industries project (Group 32).  
Covers demand forecasting, baseline simulation, GFC capacity expansion, scenario analysis, and sensitivity experiments.

---

## Requirements

Python 3.9+ with the following packages:

```
pip install pandas numpy matplotlib openpyxl
```

---

## Input files

| File | Description |
|---|---|
| `data.xlsx` | Full work orders and routings |

The file must be in the same folder as the scripts. You can also pass an explicit path as a CLI argument.

---

## How to run

### Run everything (recommended)

```bash
python run_all.py
```

This executes the full pipeline in order:

| Step | Script | What it does |
|---|---|---|
| 0 | `demand_forecast.py` | Demand forecast and capacity load analysis |
| 1 | `sdg_step1_simulation.py` | Single-workcenter validation |
| 2 | `sdg_step2_simulation.py` | Multi-workcenter routing integration |
| 3 | `main.py` | GFC-integrated baseline simulation |
| 4a | `scenario_analysis.py` | Scenario analysis (demand growth) |
| 4b | `sensitivity_analysis.py` | Sensitivity analysis (dispatch rule, uptime) |
| 4c | `plot_report.py` | Report-quality figures |

To run only specific steps:

```bash
python run_all.py data.xlsx 0 3 4a       # forecast + baseline + scenarios
python run_all.py data.xlsx 1 2          # steps 1 and 2 only
```

### Run individual scripts

```bash
python main.py                           # baseline simulation (uses data.xlsx)
python main.py "C:/path/to/data.xlsx"   # explicit data file path

python scenario_analysis.py
python sensitivity_analysis.py
python plot_report.py
```

---

## Output

All results are written to `./output/`:

```
output/
  step3/
    step3_simulation.png        — 6-panel results plot
    step3_wo_results.csv        — per-WO lead time and tardiness
    step3_op_results.csv        — per-operation queue and processing times
    step3_expansion_log.csv     — GFC capacity expansion decisions
    step3_alert_log.csv         — GFC alerts
  step4_S0/                     — baseline scenario outputs
  step4_S1/                     — conservative growth scenario outputs
  step4_S2/                     — high growth scenario outputs
  step4_comparison*.png/csv     — cross-scenario comparison plots and tables
```

---

## Configuration

All simulation parameters are in `config.py`. Edit only this file to change settings — no other script needs to be touched.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `SIM_START_STR` | `"2025-01-01"` | Simulation start date |
| `SIM_END_STR` | `"2026-12-31"` | Simulation end date |
| `WARMUP_DAYS` | `90.0` | Warm-up period excluded from metrics |
| `ROLLING_WINDOW_DAYS` | `60.0` | Window for rolling utilisation |
| `SNAP_INTERVAL` | `1.0` | Snapshot recording frequency |

---

## Module overview

```
config.py           — centralised parameters
data_loader.py      — loads work orders and routings from Excel
workcenter.py       — WorkCenter and Machine classes (O(1) idle lookup, heapq queue)
gfc.py              — GFC monitor: utilisation alerts and capacity expansion logic
events.py           — event engine: heapq event queue, EDD dispatch, simulation loop
main.py             — entry point: metrics, plots, CSV export (Step 3)
scenario_analysis.py — Step 4a: demand growth scenarios (S0/S1/S2/S6)
sensitivity_analysis.py — Step 4b: dispatch rule and uptime sensitivity (EXP A/B/C)
plot_report.py      — Step 4c: publication-quality figures
run_all.py          — master runner: executes all steps in isolated subprocesses
```

---

## Repository

Source code: [JoaneAparicio/1CM310-SI-Group-32](https://github.com/JoaneAparicio/1CM310-SI-Group-32)
