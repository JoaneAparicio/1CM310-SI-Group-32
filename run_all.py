"""
run_all.py — SDG Shopfloor Capacity Simulation
===============================================
Master entry point. Runs the full simulation pipeline in order:

  Step 0 — Demand forecast & capacity load analysis  (demand_forecast.py)
  Step 1 — Single workcenter validation              (sdg_step1_simulation.py)
  Step 2 — Multi-workcenter routing integration      (sdg_step2_simulation.py)
  Step 3 — GFC-integrated baseline simulation        (main.py logic)
  Step 4a — Scenario analysis (demand growth)        (scenario_analysis.py)
  Step 4b — Sensitivity analysis (dispatch, uptime)  (sensitivity_analysis.py)
  Step 4c — Report-quality figures                   (plot_report.py)

All outputs are written to ./output/.

Usage
-----
  python run_all.py                          # uses data.xlsx in the same folder
  python run_all.py "C:/path/to/data.xlsx"  # explicit path

To run only specific steps, pass their numbers as extra arguments:
  python run_all.py data.xlsx 0 3 4a        # only forecast, baseline, scenarios
  python run_all.py data.xlsx 1 2           # only step 1 and step 2

Notes
-----
- Steps 1 and 2 use SDG_ProcessingTime.xlsx (early build validation).
- Steps 3, 4a, 4b, 4c use data.xlsx (full dataset).
- Each step runs in an isolated subprocess so module-level state does not
  bleed between steps (the event queue, GFC logs, etc. are all reset).
- A summary of pass/fail status is printed at the end.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ── Resolve data file ──────────────────────────────────────────────────────────
args       = sys.argv[1:]
data_file  = HERE / "data.xlsx"          # default
step_filter: list[str] | None = None

# Parse args: first non-step token is the data path, rest are step IDs
step_ids_valid = {"0", "1", "2", "3", "4a", "4b", "4c"}
step_tokens    = [a for a in args if a in step_ids_valid]
path_tokens    = [a for a in args if a not in step_ids_valid]

if path_tokens:
    data_file = Path(path_tokens[0])
if step_tokens:
    step_filter = step_tokens

if not data_file.is_file():
    sys.exit(
        f"\n[ERROR] Data file not found: {data_file}\n"
        "  Place data.xlsx next to run_all.py, or pass its path as an argument.\n"
    )

proc_file = HERE / "data.xlsx"
if not proc_file.is_file():
    print(
        f"[WARNING] data.xlsx not found — Steps 1 and 2 will be skipped.\n"
        f"          Expected at: {proc_file}"
    )
    has_proc = False
else:
    has_proc = True

python = sys.executable   # use the same interpreter that launched this script

# ── Step definitions ───────────────────────────────────────────────────────────
STEPS = [
    {
        "id":     "0",
        "label":  "Step 0 — Demand Forecast & Capacity Load Analysis",
        "script": HERE / "demand_forecast.py",
        "args":   [],                  # demand_forecast reads no external args
        "needs":  None,               # no file dependency beyond imports
    },
    {
        "id":     "1",
        "label":  "Step 1 — Single Workcenter Validation (WC_AMILL_HS)",
        "script": HERE / "sdg_step1_simulation.py",
        "args":   [str(proc_file)],
        "needs":  "proc",             # requires SDG_ProcessingTime.xlsx
    },
    {
        "id":     "2",
        "label":  "Step 2 — Multi-Workcenter Routing Integration",
        "script": HERE / "sdg_step2_simulation.py",
        "args":   [str(proc_file)],
        "needs":  "proc",
    },
    {
        "id":     "3",
        "label":  "Step 3 — GFC-Integrated Baseline Simulation",
        "script": HERE / "main.py",
        "args":   [str(data_file)],
        "needs":  None,
    },
    {
        "id":     "4a",
        "label":  "Step 4a — Scenario Analysis (Demand Growth)",
        "script": HERE / "scenario_analysis.py",
        "args":   [str(data_file)],
        "needs":  None,
    },
    {
        "id":     "4b",
        "label":  "Step 4b — Sensitivity Analysis (Dispatch Rules & Uptime)",
        "script": HERE / "sensitivity_analysis.py",
        "args":   [str(data_file)],
        "needs":  None,
    },
    {
        "id":     "4c",
        "label":  "Step 4c — Report Figures",
        "script": HERE / "plot_report.py",
        "args":   [],
        "needs":  "4a+4b",            # reads CSVs produced by 4a and 4b
    },
]

# ── Run pipeline ───────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def run_step(step: dict) -> tuple[bool, float]:
    """
    Run one step as a subprocess.
    Returns (success: bool, elapsed_seconds: float).
    """
    script = step["script"]
    if not script.is_file():
        print(f"  [SKIP] Script not found: {script.name}")
        return False, 0.0

    if step["needs"] == "proc" and not has_proc:
        print(f"  [SKIP] SDG_ProcessingTime.xlsx not found — skipping {script.name}")
        return False, 0.0

    cmd = [python, str(script)] + step["args"]
    print(f"\n  Running: {' '.join(str(c) for c in cmd)}\n")

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.monotonic() - t0

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}]  {step['label']}  ({elapsed:.1f}s)")
    return ok, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║        SDG SHOPFLOOR CAPACITY SIMULATION — FULL PIPELINE            ║
╚══════════════════════════════════════════════════════════════════════╝
  Started  : {run_ts}
  Data file: {data_file}
  Steps    : {", ".join(step_filter) if step_filter else "all"}
""")

    results: list[dict] = []

    for step in STEPS:
        if step_filter and step["id"] not in step_filter:
            continue

        _header(step["label"])
        ok, elapsed = run_step(step)
        results.append({"id": step["id"], "label": step["label"],
                        "ok": ok, "elapsed": elapsed})

    # ── Summary ────────────────────────────────────────────────────────────────
    _header("PIPELINE SUMMARY")
    total = sum(r["elapsed"] for r in results)
    all_ok = all(r["ok"] for r in results)

    for r in results:
        icon = "✓" if r["ok"] else "✗"
        print(f"  {icon}  {r['id']:<4}  {r['label']:<55}  {r['elapsed']:>6.1f}s")

    print(f"\n  Total wall time : {total:.1f}s")
    print(f"  Overall status  : {'ALL STEPS PASSED' if all_ok else 'ONE OR MORE STEPS FAILED'}")
    print(f"\n  All outputs written to: {HERE / 'output'}/\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
