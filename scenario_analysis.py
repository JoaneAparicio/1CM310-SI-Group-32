"""
scenario_analysis.py — SDG Shopfloor Simulation  (Step 4)
===========================================================
Capacity analysis: baseline + demand-growth scenarios.

Three scenarios derived from the VIX demand growth trajectory established in
the conceptual design report (1CM310 Assignment 11, Section 4.1):

  S0 — Baseline        : existing data.xlsx work orders, no extra demand
  S1 — Conservative    : +30% WO load in 2026 (linear-trend forecast ≈ 72 units)
  S2 — Base (high)     : +65% WO load in 2026 (65% YoY growth ≈ 92 units) +
                         further +65% in 2027 stub (≈ 152 units annualised)

Each scenario re-runs the full discrete-event simulation from a clean state
and writes its results to output/step4_<scenario>/.

After all scenarios complete, a comparative summary table and overlay plots
are saved to output/step4_comparison*.

Usage
-----
  python scenario_analysis.py                          # uses data.xlsx next to this file
  python scenario_analysis.py "C:/path/to/data.xlsx"  # explicit path
"""

from __future__ import annotations

import copy
import importlib
import sys
import types
import warnings
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import config as _cfg

if len(sys.argv) > 1:
    _cfg.DATA_FILE = Path(sys.argv[1])

BASE_DATA = _cfg.DATA_FILE

# ── Quarterly demand profile (22 – 23 – 25 – 30 %) from conceptual design ────
QUARTERLY_WEIGHTS = [0.22, 0.23, 0.25, 0.30]

# ── New-customer synthetic routings ──────────────────────────────────────────
# Each dict mimics the routing record structure used by the simulation:
#   item_name, operation, setup_days, run_days, outside_process_fix_lt,
#   wc, leadtime, planned_qtime
#
# Processing times are calibrated for high-volume parts (small run_days/unit
# so a batch of 100 takes ~1–3 machine-days total).
#
# NCA — standard milled shaft (3-op route: AMILL → LATHE → QC)
# NCB — drilled bracket        (3-op route: MMILL → CMM  → QC)
# NCC — precision housing      (4-op route: AMILL_HS → MMILL → BW → CMM)
NEW_CUSTOMER_ROUTINGS: dict[str, list[dict]] = {
    "NC_PART_A": [
        {"item_name": "NC_PART_A", "operation": 10, "setup_days": 0.05,
         "run_days": 0.008, "outside_process_fix_lt": 0.0,
         "wc": "WC_AMILL",  "leadtime": "variable", "planned_qtime": 2.0},
        {"item_name": "NC_PART_A", "operation": 20, "setup_days": 0.03,
         "run_days": 0.006, "outside_process_fix_lt": 0.0,
         "wc": "WC_LATHE",  "leadtime": "variable", "planned_qtime": 2.0},
        {"item_name": "NC_PART_A", "operation": 30, "setup_days": 0.02,
         "run_days": 0.003, "outside_process_fix_lt": 0.0,
         "wc": "WC_QC",     "leadtime": "variable", "planned_qtime": 1.0},
    ],
    "NC_PART_B": [
        {"item_name": "NC_PART_B", "operation": 10, "setup_days": 0.04,
         "run_days": 0.010, "outside_process_fix_lt": 0.0,
         "wc": "WC_MMILL",  "leadtime": "variable", "planned_qtime": 2.0},
        {"item_name": "NC_PART_B", "operation": 20, "setup_days": 0.02,
         "run_days": 0.005, "outside_process_fix_lt": 0.0,
         "wc": "WC_CMM",    "leadtime": "variable", "planned_qtime": 1.5},
        {"item_name": "NC_PART_B", "operation": 30, "setup_days": 0.02,
         "run_days": 0.003, "outside_process_fix_lt": 0.0,
         "wc": "WC_QC",     "leadtime": "variable", "planned_qtime": 1.0},
    ],
    "NC_PART_C": [
        {"item_name": "NC_PART_C", "operation": 10, "setup_days": 0.05,
         "run_days": 0.012, "outside_process_fix_lt": 0.0,
         "wc": "WC_AMILL_HS", "leadtime": "variable", "planned_qtime": 3.0},
        {"item_name": "NC_PART_C", "operation": 20, "setup_days": 0.04,
         "run_days": 0.009, "outside_process_fix_lt": 0.0,
         "wc": "WC_MMILL",   "leadtime": "variable", "planned_qtime": 2.0},
        {"item_name": "NC_PART_C", "operation": 30, "setup_days": 0.03,
         "run_days": 0.008, "outside_process_fix_lt": 0.0,
         "wc": "WC_BW",      "leadtime": "variable", "planned_qtime": 2.0},
        {"item_name": "NC_PART_C", "operation": 40, "setup_days": 0.02,
         "run_days": 0.004, "outside_process_fix_lt": 0.0,
         "wc": "WC_CMM",     "leadtime": "variable", "planned_qtime": 1.5},
    ],
}

# New-customer release schedule: Jul 2026 → Dec 2026
# ~10 WOs/month per part type, batch size 100 units, ramp-up profile
NC_MONTHLY_SCHEDULE = {
    # (year, month): {item: n_orders}
    (2026, 7):  {"NC_PART_A": 3,  "NC_PART_B": 3,  "NC_PART_C": 2},
    (2026, 8):  {"NC_PART_A": 4,  "NC_PART_B": 4,  "NC_PART_C": 3},
    (2026, 9):  {"NC_PART_A": 5,  "NC_PART_B": 5,  "NC_PART_C": 4},
    (2026, 10): {"NC_PART_A": 6,  "NC_PART_B": 6,  "NC_PART_C": 5},
    (2026, 11): {"NC_PART_A": 7,  "NC_PART_B": 7,  "NC_PART_C": 5},
    (2026, 12): {"NC_PART_A": 8,  "NC_PART_B": 8,  "NC_PART_C": 6},
}
NC_BATCH_SIZE  = 100   # units per work order
NC_PLANNED_LT  = 35    # planned lead time in calendar days (tight for high-volume)


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "id":    "S0_Baseline",
        "label": "S0 — Baseline",
        "desc":  "Existing work orders (no additional demand).",
        "scale_2026": 0.0,   # fractional extra demand added on top of existing 2026 WOs
        "scale_2027": 0.0,
        "note":  "Current MRP data; 2026 partial WOs already included in data.xlsx.",
    },
    {
        "id":    "S1_Conservative",
        "label": "S1 — Conservative (+30 % 2026)",
        "desc":  "Linear-trend forecast: ~72 VIX units in 2026 → +30 % extra WO load.",
        "scale_2026": 0.30,
        "scale_2027": 0.0,
        "note":  "Reflects conservative demand scenario from Table 4.1 of conceptual design.",
    },
    {
        "id":    "S2_Base_High",
        "label": "S2 — Base / High (+65 % 2026, +65 % 2027)",
        "desc":  "65 % YoY growth: ~92 units in 2026, ~152 units in 2027.",
        "scale_2026": 0.65,
        "scale_2027": 0.65,
        "note":  "Stress scenario — tests whether GFC expansions can absorb sustained growth.",
    },
    # ── New customer: low-mix, high-volume ────────────────────────────────────
    # A new customer onboards in Jul 2026 with 3 high-volume part types.
    # These have their own routings (not copied from existing WOs).
    # Batch sizes ~100 units; 8–10 WOs/month from Jul 2026 onwards.
    # Routing profiles:
    #   NCA — standard milled shaft:    WC_AMILL → WC_LATHE → WC_QC
    #   NCB — drilled bracket:          WC_MMILL → WC_CMM → WC_QC
    #   NCC — precision housing:        WC_AMILL_HS → WC_MMILL → WC_BW → WC_CMM
    {
        "id":    "S6_NewCustomer",
        "label": "S6 — New Customer (low-mix, high-volume, Jul 2026)",
        "desc":  "New customer from Jul 2026: 3 high-volume part types (~100 units/batch), own routings.",
        "scale_2026": 0.0,
        "scale_2027": 0.0,
        "new_customer": True,
        "note":  "Tests capacity impact of high-volume / low-mix work on top of baseline demand.",
    },
    # ── Proactive WC_BW investment scenarios ──────────────────────────────────
    # Based on simulation evidence that WC_BW hits 97-100% utilisation and GFC
    # reactively orders Tier-2 in Sep 2025 (online Mar 2026, too late for Q4 2025).
    # Proactive strategy: commit to Tier-2 (+1 machine) on day 1 so it is online
    # in Jul 2025, before the Q4 peak.  Bridge the 6-month wait with Tier-1 (+2h).
    {
        "id":    "S3_Proactive_Baseline",
        "label": "S3 — Proactive WC_BW (Baseline demand)",
        "desc":  "Day-1 Tier-2 order for WC_BW (online Jul 2025) + Tier-1 bridge → 18 h/day. Baseline demand.",
        "scale_2026": 0.0,
        "scale_2027": 0.0,
        "proactive_bw": True,   # flag consumed by run_scenario
        "note":  "Tests whether early commitment eliminates the Q4-2025 peak and queue build-up.",
    },
    {
        "id":    "S4_Proactive_Conservative",
        "label": "S4 — Proactive WC_BW (Conservative demand)",
        "desc":  "Day-1 Tier-2 order for WC_BW (online Jul 2025) + Tier-1 bridge → 18 h/day. Conservative demand.",
        "scale_2026": 0.30,
        "scale_2027": 0.0,
        "proactive_bw": True,   # flag consumed by run_scenario
        "note":  "Tests conservative demand scenario with proactive investment strategy.",
    },
    {
        "id":    "S5_Proactive_High",
        "label": "S5 — Proactive WC_BW (Base/High demand)",
        "desc":  "Day-1 Tier-2 order for WC_BW + Tier-1 bridge. Base/High demand (+65%/+65%).",
        "scale_2026": 0.65,
        "scale_2027": 0.65,
        "proactive_bw": True,
        "note":  "Stress-tests whether proactive WC_BW investment is sufficient under high growth.",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# DEMAND AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def _build_new_customer_demand(
    df_wo_base: pd.DataFrame,
    routing_dict: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate synthetic work orders and routing entries for the new customer.

    Returns
    -------
    df_wo_aug   : df_wo_base plus new-customer WOs
    routing_aug : routing_dict plus NEW_CUSTOMER_ROUTINGS entries
    """
    rng = np.random.default_rng(seed=123)
    extra_rows: list[dict] = []
    counter = 1

    for (year, month), part_counts in NC_MONTHLY_SCHEDULE.items():
        month_start = date(year, month, 1)
        # Last day of month
        if month == 12:
            month_end = date(year, 12, 31)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)

        for item_name, n_orders in part_counts.items():
            for _ in range(n_orders):
                # Random workday within the month
                delta = (month_end - month_start).days
                for attempt in range(50):
                    d = month_start + timedelta(days=int(rng.integers(0, max(delta, 1))))
                    if d.weekday() < 5:
                        break
                due = d + timedelta(days=NC_PLANNED_LT)
                extra_rows.append({
                    "wo_num":     f"NC-WO-{counter:05d}",
                    "item_name":  item_name,
                    "start_date": pd.Timestamp(d),
                    "due_date":   pd.Timestamp(due),
                    "ord_qty":    NC_BATCH_SIZE,
                })
                counter += 1

    df_nc     = pd.DataFrame(extra_rows)
    df_wo_aug = pd.concat([df_wo_base, df_nc], ignore_index=True)

    # Inject synthetic routings so schedule_releases can find them
    routing_aug = {**routing_dict, **NEW_CUSTOMER_ROUTINGS}

    n_nc = len(df_nc)
    print(f"  [NewCustomer] Injected {n_nc} new-customer WOs "
          f"(Jul–Dec 2026, batch={NC_BATCH_SIZE}, "
          f"{len(NEW_CUSTOMER_ROUTINGS)} part types)")
    return df_wo_aug, routing_aug


def _augment_demand(
    df_wo_base: pd.DataFrame,
    routing_dict: dict,
    scale_2026: float,
    scale_2027: float,
) -> pd.DataFrame:
    """
    Augment baseline work orders by replicating (with scaled quantities) a
    random sample of existing WOs, distributed according to the 22/23/25/30%
    quarterly profile documented in the conceptual design.

    Strategy
    --------
    - Draw a sample of WOs proportional to scale (e.g. 0.65 → copy 65% of
      the 2026 WO volume).
    - Assign new start_dates within each quarter using the quarterly weights.
    - Due dates are shifted by the same planned offset (±0 days).
    - New WO numbers start at WO-EXTRA-000001 to avoid clashes.
    - Only items that have a routing entry are included (consistent with
      data_loader filtering).
    """
    rng = np.random.default_rng(seed=42)

    routed_items = set(routing_dict.keys())
    df_routable  = df_wo_base[df_wo_base["item_name"].isin(routed_items)].copy()

    extra_rows: list[dict] = []
    counter = 1

    def _quarter_start(year: int, q: int) -> date:
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return date(year, month, 1)

    def _quarter_end(year: int, q: int) -> date:
        month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        last  = {3: 31, 6: 30, 9: 30, 12: 31}[month]
        return date(year, month, last)

    def _random_workday(d0: date, d1: date) -> date:
        delta = (d1 - d0).days
        for _ in range(200):
            d = d0 + timedelta(days=int(rng.integers(0, max(delta, 1))))
            if d.weekday() < 5:
                return d
        return d0

    for year, scale in [(2026, scale_2026), (2027, scale_2027)]:
        if scale <= 0:
            continue
        # Filter WOs whose start year matches for sampling template
        sample_src_year = 2025  # always use 2025 WOs as the template population
        df_src = df_routable[df_routable["start_date"].dt.year == sample_src_year]
        if df_src.empty:
            df_src = df_routable

        n_to_add = max(1, int(round(len(df_src) * scale)))

        # Draw with replacement, weighted by quarterly profile
        q_counts = [max(1, int(round(n_to_add * w))) for w in QUARTERLY_WEIGHTS]
        # Adjust last quarter for rounding
        q_counts[-1] += n_to_add - sum(q_counts)
        q_counts[-1]  = max(0, q_counts[-1])

        for q_idx, q_n in enumerate(q_counts, start=1):
            if q_n <= 0:
                continue
            qstart = _quarter_start(year, q_idx)
            qend   = _quarter_end(year, q_idx)

            sample = df_src.sample(n=q_n, replace=True, random_state=int(rng.integers(0, 9999)))
            for _, row in sample.iterrows():
                start = _random_workday(qstart, qend)
                lt    = max(5, int((row["due_date"] - row["start_date"]).days))
                due   = start + timedelta(days=lt)
                extra_rows.append({
                    "wo_num":     f"WO-EXTRA-{counter:06d}",
                    "item_name":  row["item_name"],
                    "start_date": pd.Timestamp(start),
                    "due_date":   pd.Timestamp(due),
                    "ord_qty":    int(row["ord_qty"]),
                })
                counter += 1

    if not extra_rows:
        return df_wo_base.copy()

    df_extra = pd.DataFrame(extra_rows)
    return pd.concat([df_wo_base, df_extra], ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# FRESH SIMULATION STATE
# ═════════════════════════════════════════════════════════════════════════════

def _fresh_modules() -> tuple:
    """
    Return freshly imported (or re-initialised) simulation modules so each
    scenario starts from a clean state without cross-contamination.
    """
    # Re-import modules that carry global mutable state
    for mod in ["events", "gfc", "workcenter"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import events as EV
    import gfc    as GFC
    import workcenter as WC

    # Verify event queue is empty after fresh import
    assert len(EV._events) == 0, "Event queue not clean after re-import!"
    return EV, GFC, WC


# ═════════════════════════════════════════════════════════════════════════════
# RUN ONE SCENARIO
# ═════════════════════════════════════════════════════════════════════════════

def run_scenario(
    scenario: dict,
    df_wo_base: pd.DataFrame,
    routing_dict: dict,
) -> dict:
    """
    Build augmented demand, run the full simulation, return result dict.
    """
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario['label']}")
    print(f"  {scenario['desc']}")
    print(f"{'='*70}")

    # ── Augment demand ────────────────────────────────────────────────────────
    if scenario.get("new_customer"):
        df_wo, routing_dict = _build_new_customer_demand(df_wo_base, routing_dict)
    else:
        df_wo = _augment_demand(
            df_wo_base,
            routing_dict,
            scenario["scale_2026"],
            scenario["scale_2027"],
        )
    n_extra = len(df_wo) - len(df_wo_base)
    print(f"  Base WOs: {len(df_wo_base):,}  |  Extra WOs added: {n_extra:,}  |  Total: {len(df_wo):,}")

    # ── Fresh module state ────────────────────────────────────────────────────
    EV, GFC, WC = _fresh_modules()

    # ── Build workcenters ─────────────────────────────────────────────────────
    workcenters = WC.build_workcenters()

    # ── Proactive WC_BW investment (if flagged) ───────────────────────────────
    # Strategy: commit on day 1 to both:
    #   Tier-1  → extend WC_BW from 16 to 18 h/day (bridge, effective day 30)
    #   Tier-2  → order +1 machine (effective day 180 = ~Jul 2025)
    # This way the new machine arrives before the Q4-2025 demand peak, and the
    # GFC module is blocked from double-ordering (tier2_ordered flag set here).
    # Costs are logged manually so they appear in the expansion report.
    if scenario.get("proactive_bw"):
        import gfc as _GFC_mod
        wc_bw = workcenters["WC_BW"]

        # ── Tier-1 bridge: 16 → 18 h/day, effective in 30 days ───────────────
        t1_effective = 30.0
        wc_bw.tier1_effective_at = t1_effective
        EV.push(t1_effective, EV.EV_EXPAND_HOURS, {
            "wc": "WC_BW", "new_hours": 18, "decision_t": 0.0,
        })
        extra_h   = 18 - wc_bw.base_hours          # +2 h
        mo_cost_t1 = extra_h * wc_bw.n_machines * 1_500   # €12,000/mo (4 machines)
        _GFC_mod.expansion_log.append({
            "sim_t":             0.0,
            "date":              _cfg.SIM_ORIGIN.isoformat(),
            "wc":                "WC_BW",
            "tier":              1,
            "label":             "proactive bridge +2h",
            "old_hours":         wc_bw.base_hours,
            "new_hours":         18,
            "extra_hours_total": extra_h,
            "effective_date":    (_cfg.SIM_ORIGIN + __import__("datetime").timedelta(days=30)).isoformat(),
            "monthly_cost_eur":  mo_cost_t1,
            "capex_eur":         0,
            "annual_opex_eur":   0,
            "trigger_util":      0.0,
        })
        print(f"  [PROACTIVE] Tier-1 WC_BW: 16→18 h/day  effective day 30  "
              f"€{mo_cost_t1:,}/mo")

        # ── Tier-2: +1 machine, 6-month lead time (180 days) ─────────────────
        t2_effective = 180.0
        wc_bw.tier2_ordered      = True
        wc_bw.tier2_effective_at = t2_effective
        EV.push(t2_effective, EV.EV_EXPAND_MACHINE, {
            "wc": "WC_BW", "decision_t": 0.0,
        })
        capex       = 500_000
        annual_opex = 100_000
        _GFC_mod.expansion_log.append({
            "sim_t":             0.0,
            "date":              _cfg.SIM_ORIGIN.isoformat(),
            "wc":                "WC_BW",
            "tier":              2,
            "label":             "proactive +1 machine",
            "old_hours":         wc_bw.base_hours,
            "new_hours":         wc_bw.base_hours,
            "extra_hours_total": 0,
            "effective_date":    (_cfg.SIM_ORIGIN + __import__("datetime").timedelta(days=180)).isoformat(),
            "monthly_cost_eur":  round(annual_opex / 12),
            "capex_eur":         capex,
            "annual_opex_eur":   annual_opex,
            "trigger_util":      0.0,
        })
        print(f"  [PROACTIVE] Tier-2 WC_BW: +1 machine   effective day 180  "
              f"€{capex:,} CAPEX  €{annual_opex:,}/yr OPEX")

    # ── Schedule events ───────────────────────────────────────────────────────
    released   = EV.schedule_releases(df_wo, routing_dict)
    n_gfc_chks = EV.schedule_gfc_checks()
    print(f"  WOs released: {released:,}  |  GFC checks: {n_gfc_chks}")

    # ── Run simulation ────────────────────────────────────────────────────────
    processed = EV.run_simulation(workcenters)
    print(f"  Events processed: {processed:,}  |  WOs completed: {len(EV.wo_completed):,}")

    # ── Build result DataFrames ───────────────────────────────────────────────
    df_done = pd.DataFrame(EV.wo_completed)
    df_ops  = pd.DataFrame(EV.op_records)

    df_done = df_done[df_done["finish_sim"] >= _cfg.WARMUP_DAYS] if len(df_done) else df_done
    df_ops  = df_ops[df_ops["finish_time"]  >= _cfg.WARMUP_DAYS] if len(df_ops)  else df_ops

    # ── Key metrics ───────────────────────────────────────────────────────────
    n_comp   = len(df_done)
    n_tardy  = int(df_done["tardy"].sum())   if n_comp else 0
    pct_ota  = 100 * (1 - n_tardy / n_comp) if n_comp else 0.0
    avg_lt   = float(df_done["lead_time"].mean())   if n_comp else 0.0
    p95_lt   = float(df_done["lead_time"].quantile(0.95)) if n_comp else 0.0
    avg_tard = float(df_done.loc[df_done["tardy"], "tardiness"].mean()) if n_tardy else 0.0

    # Per-workcenter utilisation
    util_data: dict[str, dict] = {}
    for wc_name, wc in sorted(workcenters.items()):
        if wc.cap_type == "unlimited" or wc.n_machines == 0:
            continue
        sim_days = _cfg.SIM_WORKING_DAYS if wc.dept == "MANUAL" else _cfg.SIM_END_DAYS
        avail  = wc.n_machines * wc.hours_per_day * sim_days / 24.0
        busy   = sum(m.total_busy for m in wc.machines)
        util   = 100.0 * busy / avail if avail > 0 else 0.0
        wc_ops = df_ops[df_ops["wc"] == wc_name] if len(df_ops) else pd.DataFrame()
        mean_q = float(wc_ops["queue_time"].mean()) if len(wc_ops) else 0.0
        max_q  = float(wc_ops["queue_time"].max())  if len(wc_ops) else 0.0
        util_data[wc_name] = {
            "util":    util,
            "busy":    busy,
            "avail":   avail,
            "n_mach":  wc.n_machines,
            "hours":   wc.hours_per_day,
            "ops":     wc.total_ops_done,
            "mean_q":  mean_q,
            "max_q":   max_q,
        }

    # Print utilisation table
    print(f"\n  {'WC':<14} {'Mach':>5} {'Util%':>7} {'MeanQ':>8} {'MaxQ':>7}")
    print("  " + "-"*46)
    for wc_name in sorted(util_data):
        d = util_data[wc_name]
        flag = " ◄ BOTTLENECK" if d["util"] > _cfg.UTIL_ALERT else (
               " ◄ WARNING"    if d["util"] > _cfg.UTIL_WARNING else "")
        print(f"  {wc_name:<14} {d['n_mach']:>5} {d['util']:>6.1f}% "
              f"{d['mean_q']:>7.2f} {d['max_q']:>7.2f}{flag}")

    print(f"\n  On-time delivery : {pct_ota:.1f}%  ({n_comp - n_tardy}/{n_comp})")
    print(f"  Mean lead time   : {avg_lt:.1f} d   P95: {p95_lt:.1f} d")
    print(f"  Expansions       : {len(GFC.expansion_log)}   Alerts: {len(GFC.alert_log)}")

    return {
        "scenario":       scenario,
        "df_wo":          df_wo,
        "df_done":        df_done,
        "df_ops":         df_ops,
        "workcenters":    workcenters,
        "util_data":      util_data,
        "expansion_log":  GFC.expansion_log,
        "alert_log":      GFC.alert_log,
        "released":       released,
        "n_extra":        n_extra,
        "n_comp":         n_comp,
        "n_tardy":        n_tardy,
        "pct_ota":        pct_ota,
        "avg_lt":         avg_lt,
        "p95_lt":         p95_lt,
        "avg_tard":       avg_tard,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SAVE SCENARIO OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def _save_scenario_csvs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = result["scenario"]["id"]
    if len(result["df_done"]):
        result["df_done"].to_csv(out_dir / f"{sid}_wo_results.csv", index=False)
    if len(result["df_ops"]):
        result["df_ops"].to_csv(out_dir / f"{sid}_op_results.csv", index=False)
    if result["expansion_log"]:
        pd.DataFrame(result["expansion_log"]).to_csv(
            out_dir / f"{sid}_expansion_log.csv", index=False)
    if result["alert_log"]:
        pd.DataFrame(result["alert_log"]).to_csv(
            out_dir / f"{sid}_alert_log.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _sd(t: float) -> datetime:
    return datetime.combine(
        _cfg.SIM_ORIGIN + timedelta(days=int(t)), datetime.min.time())

def _fmt_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


SCENARIO_COLORS = {
    "S0_Baseline":              "steelblue",
    "S1_Conservative":          "darkorange",
    "S2_Base_High":             "crimson",
    "S3_Proactive_Baseline":    "seagreen",
    "S4_Proactive_Conservative":"mediumpurple",
    "S5_Proactive_High":        "darkviolet",
    "S6_NewCustomer":           "teal",
}
SCENARIO_LS = {
    "S0_Baseline":              "-",
    "S1_Conservative":          "--",
    "S2_Base_High":             "-.",
    "S3_Proactive_Baseline":    ":",
    "S4_Proactive_Conservative":(0, (3, 1)),
    "S5_Proactive_High":        (0, (5, 1)),
    "S6_NewCustomer":           (0, (4, 2, 1, 2)),
}


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: list[dict], out_dir: Path) -> Path:
    """
    6-panel comparison figure: utilisation bars, rolling util top WCs,
    lead-time CDFs, queue depth, on-time delivery rate, bottleneck heatmap.
    """
    fig, axes = plt.subplots(3, 2, figsize=(17, 15))
    fig.suptitle(
        "Step 4 — Capacity & Scenario Analysis  |  SDG Shopfloor Simulation\n"
        "Baseline vs Conservative (+30% 2026) vs Base/High (+65% 2026/27)",
        fontsize=12, fontweight="bold",
    )

    wc_names = sorted({wc for r in results for wc in r["util_data"]})
    s_labels = [r["scenario"]["label"] for r in results]
    s_ids    = [r["scenario"]["id"]    for r in results]
    cols     = [SCENARIO_COLORS[sid]   for sid in s_ids]

    # ── 1. Workcenter utilisation grouped bar ─────────────────────────────────
    ax = axes[0, 0]
    n_wc  = len(wc_names)
    n_sc  = len(results)
    bw    = 0.25
    x_pos = np.arange(n_wc)

    for i, (r, col) in enumerate(zip(results, cols)):
        utils = [r["util_data"].get(wc, {}).get("util", 0.0) for wc in wc_names]
        bars  = ax.bar(x_pos + i * bw, utils, width=bw, color=col,
                       alpha=0.82, label=r["scenario"]["label"], edgecolor="white")

    ax.axhline(_cfg.UTIL_ALERT,   color="red",    linestyle="--", linewidth=1,
               alpha=0.7, label=f"{_cfg.UTIL_ALERT}% alert")
    ax.axhline(_cfg.UTIL_WARNING, color="orange", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"{_cfg.UTIL_WARNING}% warning")
    ax.set_xticks(x_pos + bw)
    ax.set_xticklabels(wc_names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Utilisation %")
    ax.set_title("Workcenter Utilisation by Scenario")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25, axis="y")

    # ── 2. Rolling utilisation over time — WC_AMILL_HS (Auriga) ──────────────
    ax = axes[0, 1]
    key_wcs = ["WC_AMILL_HS", "WC_MMILL", "WC_AMILL"]   # primary bottleneck candidates
    for r in results:
        sid  = r["scenario"]["id"]
        col  = SCENARIO_COLORS[sid]
        ls   = SCENARIO_LS[sid]
        wcs  = r["workcenters"]
        for wc_name in key_wcs[:1]:   # show Auriga only for clarity
            snaps = wcs[wc_name].util_snaps if wc_name in wcs else []
            if not snaps:
                continue
            ts, us = zip(*snaps)
            ax.plot([_sd(t) for t in ts], [100 * u for u in us],
                    color=col, linestyle=ls, linewidth=1.0, alpha=0.85,
                    label=r["scenario"]["label"])
    ax.axhline(_cfg.UTIL_ALERT,   color="red",    linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(_cfg.UTIL_WARNING, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("WC_AMILL_HS (Auriga) Utilisation Over Time")
    ax.set_ylabel("% machines busy")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    _fmt_x(ax)
    ax.grid(True, alpha=0.25)

    # ── 3. Lead-time CDF ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    for r, col, ls in zip(results, cols, [SCENARIO_LS[s] for s in s_ids]):
        df = r["df_done"]
        if not len(df):
            continue
        lt_sorted = np.sort(df["lead_time"].values)
        cdf = np.arange(1, len(lt_sorted) + 1) / len(lt_sorted) * 100
        ax.plot(lt_sorted, cdf, color=col, linestyle=ls, linewidth=1.5,
                label=f"{r['scenario']['label']}  (P95={r['p95_lt']:.0f}d)")
    ax.axvline(50, color="red", linestyle=":", linewidth=1.2, label="50-day contractual LT")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("WO End-to-End Lead Time CDF")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 102)

    # ── 4. System-wide WIP over time ─────────────────────────────────────────
    ax = axes[1, 1]
    wc_for_wip = "WC_BW"   # top bottleneck — most informative for WIP trends
    for r in results:
        sid = r["scenario"]["id"]
        col = SCENARIO_COLORS[sid]
        ls  = SCENARIO_LS[sid]
        wcs = r["workcenters"]
        if wc_for_wip not in wcs:
            continue
        snaps = wcs[wc_for_wip].wip_snaps
        if not snaps:
            continue
        ts, ws = zip(*snaps)
        ws_s = pd.Series(ws).rolling(14, min_periods=1, center=True).mean()
        ax.plot([_sd(t) for t in ts], ws_s.values,
                color=col, linestyle=ls, linewidth=1.0, alpha=0.85,
                label=r["scenario"]["label"])
    ax.set_title(f"WIP at {wc_for_wip} (queue + in-processing, 14-day rolling)")
    ax.set_ylabel("Jobs (WIP)")
    ax.legend(fontsize=8)
    _fmt_x(ax)
    ax.grid(True, alpha=0.25)

    # ── 5. Rolling on-time delivery rate ──────────────────────────────────────
    ax = axes[2, 0]
    for r in results:
        sid = r["scenario"]["id"]
        col = SCENARIO_COLORS[sid]
        ls  = SCENARIO_LS[sid]
        df  = r["df_done"]
        if not len(df):
            continue
        dcs = df.sort_values("finish_sim").copy()
        dcs["finish_dt"]   = dcs["finish_sim"].apply(_sd)
        dcs["on_time"]     = (~dcs["tardy"]).astype(int)
        dcs["rolling_ota"] = dcs["on_time"].rolling(50, min_periods=1).mean() * 100
        ax.plot(dcs["finish_dt"].values, dcs["rolling_ota"].values,
                color=col, linestyle=ls, linewidth=1.2,
                label=f"{r['scenario']['label']}  ({r['pct_ota']:.0f}%)")
    ax.axhline(95,  color="red", linestyle="--", linewidth=1.2, label="95 % target")
    ax.set_ylim(0, 105)
    ax.set_title("Rolling On-Time Delivery Rate (50-WO window)")
    ax.set_ylabel("On-time %")
    ax.legend(fontsize=7)
    _fmt_x(ax)
    ax.grid(True, alpha=0.25)

    # ── 6. Bottleneck heatmap: util % per WC per scenario ────────────────────
    ax = axes[2, 1]
    heat_matrix = np.zeros((len(wc_names), len(results)))
    for j, r in enumerate(results):
        for i, wc in enumerate(wc_names):
            heat_matrix[i, j] = r["util_data"].get(wc, {}).get("util", 0.0)

    im = ax.imshow(heat_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r["scenario"]["id"].replace("_", "\n") for r in results],
                       fontsize=7)
    ax.set_yticks(range(len(wc_names)))
    ax.set_yticklabels(wc_names, fontsize=7)
    ax.set_title("Utilisation Heatmap (%) — All WCs × Scenarios")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Util %")
    for i in range(len(wc_names)):
        for j in range(len(results)):
            v = heat_matrix[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v > 65 else "black")

    plt.tight_layout()
    path = out_dir / "step4_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════════════════
# DETAILED BOTTLENECK PLOT — per scenario, top-5 WCs over time
# ═════════════════════════════════════════════════════════════════════════════

def plot_bottleneck_detail(results: list[dict], out_dir: Path) -> Path:
    """
    For each scenario, plot rolling utilisation for the 5 most-loaded WCs.
    """
    n_sc  = len(results)
    fig, axes = plt.subplots(n_sc, 1, figsize=(14, 5 * n_sc), sharex=False)
    if n_sc == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        sid  = r["scenario"]["id"]
        ud   = r["util_data"]
        wcs  = r["workcenters"]
        exp  = r["expansion_log"]

        top5 = sorted(ud, key=lambda w: ud[w]["util"], reverse=True)[:5]
        cmap = plt.cm.tab10
        for k, wc_name in enumerate(top5):
            snaps = wcs[wc_name].util_snaps if wc_name in wcs else []
            if not snaps:
                continue
            ts, us = zip(*snaps)
            us_s   = pd.Series([100 * u for u in us]).rolling(
                14, min_periods=1, center=True).mean()
            ax.plot([_sd(t) for t in ts], us_s.values,
                    color=cmap(k), linewidth=1.2, alpha=0.85,
                    label=f"{wc_name}  ({ud[wc_name]['util']:.1f}%)")

        ax.axhline(_cfg.UTIL_ALERT,   color="red",    linestyle="--", lw=1, alpha=0.7)
        ax.axhline(_cfg.UTIL_WARNING, color="orange", linestyle="--", lw=0.8, alpha=0.6)

        # Expansion vlines
        for e in exp:
            dt = datetime.combine(date.fromisoformat(e["effective_date"]),
                                  datetime.min.time())
            col = "darkgreen" if e["tier"] == 1 else "purple"
            ax.axvline(dt, color=col, linestyle=":", lw=1.4, alpha=0.8)

        ax.set_title(f"{r['scenario']['label']} — Top-5 WC Utilisation (14-day rolling)",
                     fontweight="bold", fontsize=9)
        ax.set_ylabel("Util %")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=7, loc="upper left", ncol=3, framealpha=0.7)
        _fmt_x(ax)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = out_dir / "step4_bottleneck_detail.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (CSV + printed)
# ═════════════════════════════════════════════════════════════════════════════

def build_summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        sc = r["scenario"]
        ud = r["util_data"]
        exp = r["expansion_log"]

        top_wc   = max(ud, key=lambda w: ud[w]["util"]) if ud else "–"
        top_util = ud[top_wc]["util"] if ud else 0.0
        bottlenecks = [w for w, d in ud.items() if d["util"] > _cfg.UTIL_ALERT]
        warnings_wc = [w for w, d in ud.items()
                       if _cfg.UTIL_WARNING < d["util"] <= _cfg.UTIL_ALERT]
        tier1 = [e for e in exp if e["tier"] == 1]
        tier2 = [e for e in exp if e["tier"] == 2]
        total_capex   = sum(e["capex_eur"]        for e in exp)
        total_monthly = sum(e["monthly_cost_eur"] for e in exp)

        rows.append({
            "Scenario":            sc["label"],
            "Extra WOs":           r["n_extra"],
            "WOs completed":       r["n_comp"],
            "Tardy WOs":           r["n_tardy"],
            "On-time %":           round(r["pct_ota"], 1),
            "Mean LT (d)":         round(r["avg_lt"],  1),
            "P95 LT (d)":          round(r["p95_lt"],  1),
            "Avg tardiness (d)":   round(r["avg_tard"],1),
            "Top WC":              top_wc,
            "Top WC util %":       round(top_util, 1),
            "Bottlenecks (>85%)":  ", ".join(bottlenecks) if bottlenecks else "None",
            "Warnings (>70%)":     ", ".join(warnings_wc) if warnings_wc else "None",
            "Tier-1 expansions":   len(tier1),
            "Tier-2 expansions":   len(tier2),
            "Total CAPEX (€)":     total_capex,
            "Extra monthly (€)":   total_monthly,
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*70}")
    print("  STEP 4 — CAPACITY ANALYSIS: BASELINE + DEMAND GROWTH SCENARIOS")
    print(f"{'#'*70}")
    print(f"  Data file     : {BASE_DATA}")
    print(f"  Sim horizon   : {_cfg.SIM_START_STR} → {_cfg.SIM_END_STR}")
    print(f"  Dispatch rule : {_cfg.DISPATCH_RULE}")
    print(f"  GFC enabled   : {_cfg.GFC_EXPANSIONS_ENABLED}")

    OUT    = _cfg.HERE / "output" / "scenario_comparison"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder : {OUT}")

    # ── Load data once ────────────────────────────────────────────────────────
    from data_loader import load_data
    df_wo_base, routing_dict = load_data(BASE_DATA)

    # ── Run all scenarios ─────────────────────────────────────────────────────
    results = []
    for sc in SCENARIOS:
        r = run_scenario(sc, df_wo_base, routing_dict)
        _save_scenario_csvs(r, OUT)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    df_summary = build_summary_table(results)
    summary_path = OUT / "step4_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n\n{'='*70}")
    print("  STEP 4 — COMPARATIVE SUMMARY TABLE")
    print(f"{'='*70}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df_summary.to_string(index=False))
    print(f"\n  Summary CSV → {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots…")
    p1 = plot_comparison(results, OUT)
    p2 = plot_bottleneck_detail(results, OUT)
    print(f"  Comparison plot    → {p1}")
    print(f"  Bottleneck detail  → {p2}")

    # ── Bottleneck shift analysis ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BOTTLENECK SHIFT ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  {'WC':<14}", end="")
    for r in results:
        print(f"  {r['scenario']['id'][:16]:<18}", end="")
    print()
    print("  " + "-"*70)
    all_wcs = sorted({wc for r in results for wc in r["util_data"]})
    for wc in all_wcs:
        print(f"  {wc:<14}", end="")
        for r in results:
            u = r["util_data"].get(wc, {}).get("util", 0.0)
            flag = " ◄" if u > _cfg.UTIL_ALERT else (" ~" if u > _cfg.UTIL_WARNING else "  ")
            print(f"  {u:>6.1f}%{flag}          ", end="")
        print()

    # ── Expansion cost comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CAPACITY EXPANSION COST COMPARISON")
    print(f"{'='*70}")
    for r in results:
        exp = r["expansion_log"]
        t1  = [e for e in exp if e["tier"] == 1]
        t2  = [e for e in exp if e["tier"] == 2]
        print(f"\n  {r['scenario']['label']}")
        if not exp:
            print("    No capacity expansions triggered.")
        else:
            for e in exp:
                print(f"    [{e['date']}] Tier-{e['tier']}  {e['wc']:<14}  "
                      f"effective {e['effective_date']}  "
                      f"trigger={e['trigger_util']}%  "
                      f"€{e['monthly_cost_eur']:,}/mo  "
                      f"CAPEX €{e['capex_eur']:,}")
            capex   = sum(e["capex_eur"] for e in exp)
            monthly = sum(e["monthly_cost_eur"] for e in exp)
            print(f"    Total CAPEX: €{capex:,}   Extra monthly: €{monthly:,}/mo")

    print(f"\n{'#'*70}")
    print("  Step 4 complete — all outputs written to ./output/")
    print(f"{'#'*70}\n")

    # ── Proactive vs Reactive investment comparison ───────────────────────────
    proactive_pairs = [
        ("S0_Baseline",  "S3_Proactive_Baseline",    "Baseline demand"),
        ("S1_Conservative", "S4_Proactive_Conservative", "Conservative demand"),
        ("S2_Base_High", "S5_Proactive_High",         "Base/High demand"),
    ]
    res_by_id = {r["scenario"]["id"]: r for r in results}
    print(f"\n{'='*70}")
    print("  PROACTIVE vs REACTIVE WC_BW INVESTMENT — HEAD-TO-HEAD")
    print(f"{'='*70}")
    for reactive_id, proactive_id, label in proactive_pairs:
        if reactive_id not in res_by_id or proactive_id not in res_by_id:
            continue
        rr = res_by_id[reactive_id]
        rp = res_by_id[proactive_id]
        bw_r = rr["util_data"].get("WC_BW", {})
        bw_p = rp["util_data"].get("WC_BW", {})
        capex_r = sum(e["capex_eur"] for e in rr["expansion_log"])
        capex_p = sum(e["capex_eur"] for e in rp["expansion_log"])
        mo_r    = sum(e["monthly_cost_eur"] for e in rr["expansion_log"])
        mo_p    = sum(e["monthly_cost_eur"] for e in rp["expansion_log"])
        print(f"\n  [{label}]")
        print(f"  {'Metric':<30} {'Reactive':>14} {'Proactive':>14}  {'Delta':>10}")
        print("  " + "-"*72)
        def _row(name, rv, pv, fmt="{:.1f}"):
            delta = pv - rv if isinstance(rv, (int, float)) else ""
            fv = fmt.format
            sign = "+" if isinstance(delta, float) and delta > 0 else ""
            delta_s = f"{sign}{fmt.format(delta)}" if isinstance(delta, float) else "–"
            print(f"  {name:<30} {fmt.format(rv):>14} {fmt.format(pv):>14}  {delta_s:>10}")
        _row("WC_BW util % (overall)",  bw_r.get("util",0),   bw_p.get("util",0))
        _row("WC_BW max queue (days)",  bw_r.get("max_q",0),  bw_p.get("max_q",0))
        _row("WC_BW mean queue (days)", bw_r.get("mean_q",0), bw_p.get("mean_q",0))
        _row("On-time delivery %",      rr["pct_ota"],         rp["pct_ota"])
        _row("Mean lead time (d)",      rr["avg_lt"],          rp["avg_lt"])
        _row("P95 lead time (d)",       rr["p95_lt"],          rp["p95_lt"])
        _row("Total CAPEX (€)",         float(capex_r),        float(capex_p), "{:.0f}")
        _row("Extra monthly cost (€)",  float(mo_r),           float(mo_p),    "{:.0f}")
        print(f"\n  Machine online (reactive) : GFC-triggered ~Sep 2025 → online Mar 2026")
        print(f"  Machine online (proactive): day-1 order → online Jul 2025 (8 months earlier)")



if __name__ == "__main__":
    main()