"""
plot_wc_utilisation.py — SDG Shopfloor Simulation
===================================================
Generates a figure with one subplot per capacity-constrained workcenter.
Each subplot shows the rolling utilisation over time for every individual
machine in that workcenter, plus the workcenter aggregate (average).

Run AFTER main.py (needs the op_records CSV produced by the simulation):
    python plot_wc_utilisation.py
    python plot_wc_utilisation.py "C:/path/to/data.xlsx"   # same arg as main.py

Output
------
    output/wc_utilisation_per_machine.png
"""

import sys
from datetime import date, timedelta, datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config

if len(sys.argv) > 1:
    config.DATA_FILE = Path(sys.argv[1])

config.OUT.mkdir(parents=True, exist_ok=True)

# ── Load op-level results produced by main.py ────────────────────────────────
CSV_PATH = config.OUT / "step3" / "step3_op_results.csv"
if not CSV_PATH.is_file():
    sys.exit(
        f"\n[ERROR] {CSV_PATH} not found.\n"
        "  Run main.py first to generate the simulation results.\n"
    )

df_ops = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_ops):,} operation records from {CSV_PATH}")

# ── Load expansion log if available (for vlines) ─────────────────────────────
EXP_PATH = config.OUT / "step3" / "step3_expansion_log.csv"
expansion_log = (
    pd.read_csv(EXP_PATH).to_dict("records") if EXP_PATH.is_file() else []
)

# ── Helpers ───────────────────────────────────────────────────────────────────
SIM_ORIGIN   = config.SIM_ORIGIN
SIM_END_DAYS = config.SIM_END_DAYS

def sim_to_dt(t: float) -> datetime:
    return datetime.combine(SIM_ORIGIN + timedelta(days=int(t)), datetime.min.time())

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

def _add_expansion_vlines(ax, wc_name: str):
    """Overlay expansion effective-date lines for this WC only."""
    for e in expansion_log:
        if e["wc"] == wc_name:
            dt = datetime.combine(
                date.fromisoformat(e["effective_date"]), datetime.min.time()
            )
            tier = e["tier"]
            color = "darkgreen" if tier == 1 else "purple"
            label = f"Tier-{tier} effective"
            ax.axvline(dt, color=color, linestyle=":", linewidth=1.4,
                       alpha=0.85, label=label)

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PER-MACHINE DAILY UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════
# Strategy: for each operation record we know start_time and finish_time
# (both in sim-days).  We spread the busy time across calendar days by
# intersecting the operation interval with each day bucket [d, d+1).
# Then utilisation on day d = busy_hours / available_hours_per_day.

def compute_daily_utilisation(
    df: pd.DataFrame,
    wc_name: str,
    n_machines: int,
    hours_per_day: float,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        day_sim  : float (integer sim-day index)
        machine  : int   (0-based machine id, or 'aggregate')
        util_pct : float (0-100)

    Machine IDs are inferred from the order jobs were processed
    (we don't have explicit machine_id in op_records, so we use a
    simple heuristic: assign ops to machines greedily in start_time order,
    packing them into the machine that finishes earliest — same as the sim).
    """
    sub = df[df["wc"] == wc_name].copy()
    if sub.empty:
        return pd.DataFrame(columns=["day_sim", "machine", "util_pct"])

    sub = sub.sort_values("start_time").reset_index(drop=True)

    # Greedy machine assignment (mirrors simulation logic)
    machine_free = [0.0] * n_machines   # earliest free time per machine
    machine_ids  = []
    for _, row in sub.iterrows():
        # Pick machine that is free earliest
        mid = int(np.argmin(machine_free))
        machine_ids.append(mid)
        machine_free[mid] = row["finish_time"]
    sub["machine_id"] = machine_ids

    # Build day grid
    n_days = int(SIM_END_DAYS) + 1
    # busy_per_machine[m][d] = busy days on calendar-day d
    busy = np.zeros((n_machines, n_days), dtype=float)

    for _, row in sub.iterrows():
        mid = int(row["machine_id"])
        t0  = float(row["start_time"])
        t1  = float(row["finish_time"])
        d0  = int(t0)
        d1  = min(int(t1) + 1, n_days)
        for d in range(d0, d1):
            overlap = min(t1, d + 1) - max(t0, d)
            if overlap > 0:
                busy[mid, d] += overlap

    avail_per_day = hours_per_day / 24.0   # fraction of a calendar day

    records = []
    day_range = np.arange(n_days)
    for m in range(n_machines):
        util = np.clip(busy[m] / avail_per_day * 100.0, 0, 100)
        for d, u in zip(day_range, util):
            records.append({"day_sim": float(d), "machine": m, "util_pct": u})

    # Aggregate: average across machines per day
    agg_busy = busy.sum(axis=0)
    total_avail = n_machines * avail_per_day
    agg_util = np.clip(agg_busy / total_avail * 100.0, 0, 100)
    for d, u in zip(day_range, agg_util):
        records.append({"day_sim": float(d), "machine": "agg", "util_pct": u})

    return pd.DataFrame(records)


ROLLING_DAYS = 14   # smooth with 14-day rolling average for readability

def smooth(series: pd.Series, window: int = ROLLING_DAYS) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════
# Only limited workcenters
limited_wcs = {
    wc: cfg for wc, cfg in config.WC_CONFIG.items()
    if cfg[3] == "limited" and cfg[0] is not None
}
wc_names = sorted(limited_wcs.keys())
n_wcs    = len(wc_names)

# Grid layout: 3 columns
NCOLS = 3
NROWS = (n_wcs + NCOLS - 1) // NCOLS

fig, axes = plt.subplots(
    NROWS, NCOLS,
    figsize=(7 * NCOLS, 4.5 * NROWS),
    sharex=False,
)
fig.suptitle(
    "Per-Workcenter Machine Utilisation Over Time\n"
    f"(14-day rolling average  |  dispatch: {config.DISPATCH_RULE})",
    fontsize=13, fontweight="bold", y=1.01,
)
axes_flat = axes.flatten()

for idx, wc_name in enumerate(wc_names):
    ax = axes_flat[idx]
    n_mach, dept, hpd, _ = limited_wcs[wc_name]

    # Compute daily utilisation
    df_util = compute_daily_utilisation(df_ops, wc_name, n_mach, hpd)

    if df_util.empty:
        ax.set_title(f"{wc_name}\n(no data)")
        ax.axis("off")
        continue

    # Convert sim-day → datetime
    df_util["date"] = df_util["day_sim"].apply(sim_to_dt)

    # ── Plot individual machines (thin, low alpha) ────────────────────────────
    machine_ids = sorted(
        [m for m in df_util["machine"].unique() if m != "agg"]
    )
    cmap   = plt.cm.tab20 if n_mach > 10 else plt.cm.tab10
    colors = [cmap(i / max(n_mach, 1)) for i in range(n_mach)]

    for m, col in zip(machine_ids, colors):
        sub = df_util[df_util["machine"] == m].sort_values("day_sim")
        s   = smooth(sub["util_pct"])
        ax.plot(sub["date"].values, s.values,
                color=col, linewidth=0.7, alpha=0.5,
                label=f"M{m}")

    # ── Plot aggregate (thick, solid) ─────────────────────────────────────────
    agg = df_util[df_util["machine"] == "agg"].sort_values("day_sim")
    s_agg = smooth(agg["util_pct"])
    ax.plot(agg["date"].values, s_agg.values,
            color="black", linewidth=2.2, alpha=0.9,
            label="Average", zorder=5)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(config.UTIL_ALERT,   color="red",    linestyle="--", linewidth=1,   alpha=0.8,
               label=f"{config.UTIL_ALERT}% alert")
    ax.axhline(config.UTIL_WARNING, color="orange", linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"{config.UTIL_WARNING}% warning")

    # ── Expansion vlines ──────────────────────────────────────────────────────
    _add_expansion_vlines(ax, wc_name)

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_title(
        f"{wc_name}  ({n_mach} machines, {dept}, {hpd} h/day)",
        fontsize=9, fontweight="bold",
    )
    ax.set_ylabel("Utilisation %", fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_xlim(
        sim_to_dt(0),
        sim_to_dt(SIM_END_DAYS),
    )
    _fmt_xaxis(ax)
    ax.grid(True, alpha=0.25)

    # Legend: only show aggregate + thresholds (suppress individual machines
    # unless there are ≤ 4, to keep it readable)
    handles, labels = ax.get_legend_handles_labels()
    if n_mach <= 4:
        ax.legend(handles, labels, fontsize=6, loc="upper left",
                  ncol=2, framealpha=0.7)
    else:
        # Show only Average + reference lines
        keep = [(h, l) for h, l in zip(handles, labels)
                if l in ("Average", "85% alert", "70% warning",
                          "Tier-1 effective", "Tier-2 effective")]
        if keep:
            hs, ls = zip(*keep)
            ax.legend(hs, ls, fontsize=7, loc="upper left", framealpha=0.7)

# ── Hide unused subplots ──────────────────────────────────────────────────────
for idx in range(n_wcs, len(axes_flat)):
    axes_flat[idx].axis("off")

plt.tight_layout()
suffix = "with_expansion" if config.GFC_EXPANSIONS_ENABLED else "no_expansion"
plot_path = config.OUT / f"wc_utilisation_{suffix}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")
