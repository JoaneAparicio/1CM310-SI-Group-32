"""
main.py — SDG Shopfloor Simulation
=====================================
Entry point.  Orchestrates all modules:

  1. config.py      — parameters (edit here to change any setting)
  2. data_loader.py — load work orders and routings from data.xlsx
  3. workcenter.py  — instantiate WorkCenter / Machine objects
  4. gfc.py         — GFC monitoring and capacity expansion logic
  5. events.py      — event engine and simulation loop
  6. main.py        — metrics, plots, CSV export (this file)

Usage
-----
  python main.py                              # uses data.xlsx next to this file
  python main.py "C:/path/to/data.xlsx"       # explicit path

Output (written to ./output/)
------------------------------
  step3_simulation.png   — 6-panel results plot
  step3_wo_results.csv   — per-WO lead time and tardiness
  step3_op_results.csv   — per-operation queue and processing times
  step3_expansion_log.csv — capacity expansion decisions
  step3_alert_log.csv    — GFC alerts
"""

import sys
from datetime import date, timedelta, datetime
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Override DATA_FILE from CLI before importing config ───────────────────────
import config
if len(sys.argv) > 1:
    config.DATA_FILE = Path(sys.argv[1])

HERE = Path(__file__).parent.resolve()
OUT = HERE / "output" / "step3"
config.OUT.mkdir(parents=True, exist_ok=True)

# ── Tee: redirige stdout a pantalla + fichero de log ─────────────────────────
class _Tee:
    def __init__(self, file_stream):
        self._file    = file_stream
        self._console = sys.__stdout__
    def write(self, data):
        self._console.write(data)
        self._console.flush()
        self._file.write(data)
        self._file.flush()
    def flush(self):
        self._console.flush()
        self._file.flush()

_run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = config.OUT / f"results_{_run_ts}.txt"
_log_file = open(_log_path, "w", encoding="utf-8")
_log_file.write(f"Run started: {datetime.now().isoformat()}\n{'='*70}\n\n")
_log_file.flush()
sys.stdout = _Tee(_log_file)
# ── A partir de aquí TODO print() va también al .txt ─────────────────────────

if not config.DATA_FILE.is_file():
    sys.exit(
        f"\n[ERROR] File not found: {config.DATA_FILE}\n"
        "  Pass the full path as argument or place data.xlsx next to main.py.\n"
    )

print(f"Data file : {config.DATA_FILE}")

# ── Module imports (after config path is set) ─────────────────────────────────
from data_loader import load_data
from workcenter  import build_workcenters
import gfc    as GFC
import events as EV

# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
df_wo, routing_dict = load_data(config.DATA_FILE)

# ═════════════════════════════════════════════════════════════════════════════
# 2.  BUILD WORKCENTERS
# ═════════════════════════════════════════════════════════════════════════════
workcenters = build_workcenters()

# ═════════════════════════════════════════════════════════════════════════════
# 3.  SCHEDULE INITIAL EVENTS
# ═════════════════════════════════════════════════════════════════════════════
released   = EV.schedule_releases(df_wo, routing_dict)
n_gfc_chks = EV.schedule_gfc_checks()
print(f"Work orders scheduled : {released}")
print(f"GFC check events      : {n_gfc_chks}")

# ═════════════════════════════════════════════════════════════════════════════
# 4.  RUN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning simulation…")
processed = EV.run_simulation(workcenters)
print(f"Simulation done. Events processed: {processed}")
print(f"WOs completed: {len(EV.wo_completed)} / {released}")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  BUILD RESULT DATAFRAMES
# ═════════════════════════════════════════════════════════════════════════════
df_wo_done = pd.DataFrame(EV.wo_completed)
df_ops     = pd.DataFrame(EV.op_records)
df_wo_done = df_wo_done[df_wo_done["finish_sim"] >= config.WARMUP_DAYS]
df_ops     = df_ops[df_ops["finish_time"]        >= config.WARMUP_DAYS]
# ═════════════════════════════════════════════════════════════════════════════
# 6.  METRICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  STEP 3 RESULTS  |  dispatch: {config.DISPATCH_RULE}  |  GFC active")
print("=" * 70)

if len(df_wo_done):
    n_tardy = int(df_wo_done["tardy"].sum())
    pct_ota = 100 * (1 - n_tardy / len(df_wo_done))
    avg_lt  = df_wo_done["lead_time"].mean()
    p95_lt  = df_wo_done["lead_time"].quantile(0.95)
    print(f"\n  WOs completed     : {len(df_wo_done)} / {released}")
    print(f"  Mean lead time    : {avg_lt:.2f} days")
    print(f"  P95  lead time    : {p95_lt:.2f} days")
    print(f"  On-time delivery  : {pct_ota:.1f}%  "
          f"({len(df_wo_done) - n_tardy}/{len(df_wo_done)})")
    print(f"  Tardy WOs         : {n_tardy}")
    if n_tardy:
        print(f"  Avg tardiness     : "
              f"{df_wo_done.loc[df_wo_done['tardy'], 'tardiness'].mean():.2f} days")
else:
    pct_ota = avg_lt = p95_lt = 0.0

# ── Per-workcenter utilisation table ─────────────────────────────────────────
print(f"\n  {'WC':<14} {'Mach':>5} {'Hrs':>4} {'Avail(d)':>10} {'Busy(d)':>10} "
      f"{'Util%':>7} {'Ops':>6} {'MeanQ(d)':>9} {'MaxQ(d)':>8}")
print("  " + "-" * 72)

util_data: dict[str, dict] = {}
for wc_name, wc in sorted(workcenters.items()):
    if wc.cap_type == "unlimited" or wc.n_machines == 0:
        continue

    # MANUAL WCs are only staffed Mon–Fri → use working days as denominator.
    # AUTO WCs can run through weekends (jobs continue after Friday) → calendar days.
    sim_days = config.SIM_WORKING_DAYS if wc.dept == "MANUAL" else config.SIM_END_DAYS
    avail  = wc.n_machines * wc.hours_per_day * sim_days / 24.0
    busy   = sum(m.total_busy for m in wc.machines)
    util   = 100.0 * busy / avail if avail > 0 else 0.0

    wc_ops = df_ops[df_ops["wc"] == wc_name] if len(df_ops) else pd.DataFrame()
    mean_q = wc_ops["queue_time"].mean() if len(wc_ops) else 0.0
    max_q  = wc_ops["queue_time"].max()  if len(wc_ops) else 0.0

    util_data[wc_name] = {
        "util": util, "busy": busy, "avail": avail,
        "ops":  wc.total_ops_done, "mean_q": mean_q, "max_q": max_q,
        "hours": wc.hours_per_day,
    }
    flag = " ◄ BOTTLENECK" if util > config.UTIL_ALERT else (" ◄ WARNING" if util > config.UTIL_WARNING else "")
    print(
        f"  {wc_name:<14} {wc.n_machines:>5} {wc.hours_per_day:>4} "
        f"{avail:>10.1f} {busy:>10.2f} {util:>6.1f}% "
        f"{wc.total_ops_done:>6} {mean_q:>9.2f} {max_q:>8.2f}{flag}"
    )

# ── GFC summary ───────────────────────────────────────────────────────────────
expansion_log = GFC.expansion_log
alert_log     = GFC.alert_log

print(f"\n  ── GFC Capacity Expansion Decisions "
      f"({'none' if not expansion_log else len(expansion_log)}) ──")
if expansion_log:
    total_monthly = total_capex = 0
    for e in expansion_log:
        print(
            f"  {e['date']}  Tier-{e['tier']}  {e['wc']:<14}  "
            f"effective {e['effective_date']}  "
            f"trigger util {e['trigger_util']}%  "
            f"€{e['monthly_cost_eur']:,}/mo  CAPEX €{e['capex_eur']:,}"
        )
        total_monthly += e["monthly_cost_eur"]
        total_capex   += e["capex_eur"]
    print(f"\n  Total extra monthly cost : €{total_monthly:,}/mo")
    print(f"  Total CAPEX              : €{total_capex:,}")
else:
    print("  No capacity expansions triggered during simulation horizon.")

print(f"\n  ── GFC Alerts fired: {len(alert_log)} ──")
for a in alert_log:
    print(f"  {a['date']}  {a['wc']:<14}  rolling util = {a['util']}%")

# ═════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def sd(t: float) -> datetime:
    """sim-days → datetime for matplotlib axes."""
    return datetime.combine(
        config.SIM_ORIGIN + timedelta(days=int(t)),
        datetime.min.time(),
    )


fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle(
    f"Step 3 — GFC Integrated  |  Dispatch: {config.DISPATCH_RULE}\n"
    f"WOs completed: {len(df_wo_done)}/{released}   "
    f"On-time: {pct_ota:.1f}%   "
    f"Mean LT: {avg_lt:.1f} d   "
    f"Expansions: {len(expansion_log)}",
    fontsize=11, fontweight="bold",
)

wc_sorted = sorted(util_data.keys())
top3      = sorted(util_data, key=lambda w: util_data[w]["util"], reverse=True)[:3]
pal       = ["red", "darkorange", "steelblue"]

def _add_expansion_vlines(ax):
    """Overlay green vertical lines for each expansion effective date."""
    for e in expansion_log:
        ax.axvline(
            datetime.combine(date.fromisoformat(e["effective_date"]), datetime.min.time()),
            color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7,
        )

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# ── 1. Utilisation bar chart ──────────────────────────────────────────────────
ax = axes[0, 0]
utils  = [util_data[w]["util"] for w in wc_sorted]
colors = ["red" if u > config.UTIL_ALERT else ("orange" if u > config.UTIL_WARNING else "steelblue") for u in utils]
bars   = ax.barh(wc_sorted, utils, color=colors, edgecolor="white", height=0.6)
ax.axvline(config.UTIL_ALERT,   color="red",    linestyle="--", linewidth=1,   label=f"{config.UTIL_ALERT}% alert")
ax.axvline(config.UTIL_WARNING, color="orange", linestyle="--", linewidth=0.8, label=f"{config.UTIL_WARNING}% warning")
ax.set_title("Workcenter Utilisation (%) — full horizon")
ax.set_xlabel("Utilisation %")
ax.set_xlim(0, 110)
ax.legend(fontsize=8)
for bar, u in zip(bars, utils):
    ax.text(u + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{u:.1f}%", va="center", fontsize=7)
expanded_wcs = {e["wc"] for e in expansion_log}
for i, wc_name in enumerate(wc_sorted):
    if wc_name in expanded_wcs:
        ax.text(3, i, "⬆ expanded", va="center", fontsize=7, color="darkgreen")
ax.grid(True, alpha=0.3, axis="x")

# ── 2. Rolling utilisation over time — top-3 WCs ─────────────────────────────
ax = axes[0, 1]
for wc_name, col in zip(top3, pal):
    snaps = workcenters[wc_name].util_snaps
    if not snaps:
        continue
    ts, us = zip(*snaps)
    ax.plot([sd(t) for t in ts], [100 * u for u in us],
            color=col, linewidth=0.8, alpha=0.8, label=wc_name)
ax.axhline(config.UTIL_ALERT,   color="red",     linestyle="--", linewidth=1,
           label=f"{config.UTIL_ALERT}% alert")
ax.axhline(config.UTIL_WARNING, color="orange",  linestyle="--", linewidth=0.8,
           label=f"{config.UTIL_WARNING}% warning")
_add_expansion_vlines(ax)
ax.set_title("Instantaneous Utilisation — Top-3 WCs")
ax.set_ylabel("% machines busy")
ax.set_ylim(0, 110)
ax.legend(fontsize=7)
_fmt_xaxis(ax)
ax.grid(True, alpha=0.3)

# ── 3. WO lead time distribution ─────────────────────────────────────────────
ax = axes[1, 0]
if len(df_wo_done):
    ax.hist(df_wo_done["lead_time"], bins=50, color="slateblue",
            edgecolor="white", linewidth=0.3)
    ax.axvline(avg_lt, color="red",        linestyle="--", label=f"Mean={avg_lt:.1f}d")
    ax.axvline(p95_lt, color="darkorange", linestyle="--", label=f"P95={p95_lt:.1f}d")
    ax.legend(fontsize=8)
ax.set_title("WO End-to-End Lead Time Distribution")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.3)

# ── 4. WIP over time — top-3 WCs ─────────────────────────────────────────────
ax = axes[1, 1]
for wc_name, col in zip(top3, pal):
    snaps = workcenters[wc_name].wip_snaps
    if not snaps:
        continue
    ts, ws = zip(*snaps)
    ws_s = pd.Series(ws).rolling(14, min_periods=1, center=True).mean()
    ax.plot([sd(t) for t in ts], ws_s.values,
            color=col, linewidth=0.9, alpha=0.85, label=wc_name)
_add_expansion_vlines(ax)
ax.set_title("WIP Over Time — Top-3 WCs  (queue + in-processing, 14-day avg)")
ax.set_ylabel("Jobs (WIP)")
ax.legend(fontsize=8)
_fmt_xaxis(ax)
ax.grid(True, alpha=0.3)

# ── 5. Mean queue time: simulated vs MRP planned ─────────────────────────────
ax = axes[2, 0]
mean_qs   = [util_data[w]["mean_q"] for w in wc_sorted]
planned_q = [
    (df_ops[df_ops["wc"] == w]["planned_q"].mean() if len(df_ops) and w in df_ops["wc"].values else 0.0)
    for w in wc_sorted
]
x = range(len(wc_sorted))
ax.bar(x,               mean_qs,   width=0.4, label="Simulated",    color="steelblue", alpha=0.8)
ax.bar([i + 0.4 for i in x], planned_q, width=0.4, label="MRP planned", color="seagreen",  alpha=0.8)
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(wc_sorted, rotation=45, ha="right", fontsize=7)
ax.set_title("Mean Queue Time: Simulated vs MRP Planned (days)")
ax.set_ylabel("Days")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ── 6. Rolling on-time delivery rate ─────────────────────────────────────────
ax = axes[2, 1]
if len(df_wo_done):
    dcs = df_wo_done.sort_values("finish_sim").copy()
    dcs["finish_dt"]   = dcs["finish_sim"].apply(sd)
    dcs["on_time"]     = (~dcs["tardy"]).astype(int)
    dcs["rolling_ota"] = dcs["on_time"].rolling(50, min_periods=1).mean() * 100
    ax.plot(dcs["finish_dt"].values, dcs["rolling_ota"].values,
            color="seagreen", linewidth=1)
    ax.axhline(95,      color="red",  linestyle="--", linewidth=1, label="95% target")
    ax.axhline(pct_ota, color="navy", linestyle=":",  linewidth=1,
               label=f"Overall {pct_ota:.1f}%")
    _add_expansion_vlines(ax)
ax.set_ylim(0, 105)
ax.set_title("Rolling On-Time Delivery Rate (50-WO window)")
ax.set_ylabel("On-time %")
ax.legend(fontsize=8)
_fmt_xaxis(ax)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = config.OUT / "step3_simulation.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 8.  SAVE CSVs
# ═════════════════════════════════════════════════════════════════════════════
if len(df_wo_done):
    p = config.OUT / "step3_wo_results.csv"
    df_wo_done.to_csv(p, index=False)
    print(f"WO results    → {p}")

if len(df_ops):
    p = config.OUT / "step3_op_results.csv"
    df_ops.to_csv(p, index=False)
    print(f"Op results    → {p}")

if expansion_log:
    pd.DataFrame(expansion_log).to_csv(config.OUT / "step3_expansion_log.csv", index=False)
    print(f"Expansion log → {config.OUT / 'step3_expansion_log.csv'}")

if alert_log:
    pd.DataFrame(alert_log).to_csv(config.OUT / "step3_alert_log.csv", index=False)
    print(f"Alert log     → {config.OUT / 'step3_alert_log.csv'}")

# ═════════════════════════════════════════════════════════════════════════════
# 9.  CONCEPTUAL DESIGN VALIDATION SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CONCEPTUAL DESIGN VALIDATION — STEP 3")
print("=" * 70)

bottlenecks = [w for w, d in util_data.items() if d["util"] > config.UTIL_ALERT]
warnings    = [w for w, d in util_data.items() if config.UTIL_WARNING < d["util"] <= config.UTIL_ALERT]
top_wc      = max(util_data, key=lambda w: util_data[w]["util"]) if util_data else "N/A"

tier1 = [e for e in expansion_log if e["tier"] == 1]
tier2 = [e for e in expansion_log if e["tier"] == 2]
total_capex   = sum(e["capex_eur"]        for e in expansion_log)
total_monthly = sum(e["monthly_cost_eur"] for e in expansion_log)

amill_hs_u = util_data.get("WC_AMILL_HS", {}).get("util", 0.0)
mmill_u    = util_data.get("WC_MMILL",    {}).get("util", 0.0)

print(f"""
  [B1] WC_AMILL_HS (Auriga) utilisation = {amill_hs_u:.1f}%
       {"→ CONFIRMED bottleneck candidate: alert expected"
        if amill_hs_u > config.UTIL_ALERT
        else "→ NOT a bottleneck at this demand level"}

  [B2] On-time delivery with EDD = {pct_ota:.1f}%
       (Re-run with DISPATCH_RULE='FIFO' or 'SPT' in config.py for comparison)

  [B3] Tier-1 expansions (hours): {len(tier1)} — {[e['wc'] for e in tier1]}
       Tier-2 expansions (machine): {len(tier2)} — {[e['wc'] for e in tier2]}
       Total CAPEX: €{total_capex:,}   Extra opex: €{total_monthly:,}/mo
       {"→ Tier-1 sufficient: no Tier-2 machines added"
        if not tier2
        else "→ Tier-2 triggered: hours-extension alone insufficient for sustained load"}

  [B4] GFC alert sensitivity:
       Alerts fired: {len(alert_log)} (threshold {config.UTIL_ALERT}%)
       Earliest alert: {alert_log[0]['date'] if alert_log else 'N/A'} \
on {alert_log[0]['wc'] if alert_log else 'N/A'}
       {"→ GFC lead time sufficient for Tier-1 notice window"
        if alert_log else "→ No alerts; all WCs within guideline"}

  Most loaded WC : {top_wc}  ({util_data.get(top_wc, {}).get('util', 0):.1f}%)
  Bottlenecks (>85%) : {bottlenecks if bottlenecks else 'None'}
  Warnings    (>70%) : {warnings   if warnings   else 'None'}
  WC_MMILL utilisation: {mmill_u:.1f}%  \
{"→ secondary bottleneck" if mmill_u > 70 else ""}

  Next step (Step 4):
      Add scenario analysis (demand growth), sensitivity (90% uptime),
      and comparative dispatching-rule experiments.
""")

# ── Cerrar log ────────────────────────────────────────────────────────────────
_log_file.write(f"\n{'='*70}\nRun finished: {datetime.now().isoformat()}\n")
_log_file.close()
sys.stdout = sys.__stdout__
sys.__stdout__.write(f"\nLog saved → {_log_path}\n")
