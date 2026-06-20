"""
sensitivity_analysis.py — SDG Shopfloor Simulation  (Step 4 extension)
========================================================================
Experiments on three design parameters, each isolated from the others
so the effect of each change is clearly attributable:

  EXP-A  Dispatching rule comparison
         A0: EDD  (current baseline)
         A1: SLACK = due_sim − now − remaining_proc_time  (lower is more urgent)
         A2: SPT  (shortest processing time first)
         A3: FIFO (arrival order — control case)

  EXP-B  GFC alert threshold sensitivity for AUTO workcenters (Auriga)
         B0: ALERT_SUSTAIN = 7 d  (current)
         B1: ALERT_SUSTAIN = 3 d  (faster trigger for AUTO WCs)
         B2: ALERT_SUSTAIN = 1 d  (near-immediate trigger)

  EXP-C  Machine availability / uptime sensitivity  (assignment requirement)
         C0: 100% uptime  (current — no unplanned downtime)
         C1:  95% uptime  (light maintenance losses)
         C2:  90% uptime  (industry-standard assumption, explicitly requested)
         C3:  85% uptime  (stress test)

All experiments run on the BASELINE demand (S0) so the comparison is clean.
Results are written to output/sensitivity_*.csv and a multi-panel figure.

Usage
-----
  python sensitivity_analysis.py
  python sensitivity_analysis.py "C:/path/to/data.xlsx"
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config as _cfg
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

if len(sys.argv) > 1:
    _cfg.DATA_FILE = Path(sys.argv[1])


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

DISPATCH_EXPERIMENTS = [
    {"id": "A0_EDD",  "label": "A0 — EDD (baseline)",  "rule": "EDD"},
    {"id": "A1_SLACK","label": "A1 — Slack",            "rule": "SLACK"},
    {"id": "A2_SPT",  "label": "A2 — SPT",              "rule": "SPT"},
    {"id": "A3_FIFO", "label": "A3 — FIFO",             "rule": "FIFO"},
]

GFC_EXPERIMENTS = [
    {"id": "B0_7d", "label": "B0 — Alert sustain 7 d (baseline)", "alert_sustain": 7.0,  "warn_sustain": 30.0},
    {"id": "B1_3d", "label": "B1 — Alert sustain 3 d",            "alert_sustain": 3.0,  "warn_sustain": 14.0},
    {"id": "B2_1d", "label": "B2 — Alert sustain 1 d",            "alert_sustain": 1.0,  "warn_sustain": 5.0},
]

UPTIME_EXPERIMENTS = [
    {"id": "C0_100", "label": "C0 — 100% uptime (baseline)", "uptime": 1.00},
    {"id": "C1_95",  "label": "C1 — 95% uptime",             "uptime": 0.95},
    {"id": "C2_90",  "label": "C2 — 90% uptime",             "uptime": 0.90},
    {"id": "C3_85",  "label": "C3 — 85% uptime",             "uptime": 0.85},
]


# ═════════════════════════════════════════════════════════════════════════════
# FRESH MODULE LOADER
# ═════════════════════════════════════════════════════════════════════════════

def _fresh() -> tuple:
    for mod in ["events", "gfc", "workcenter"]:
        if mod in sys.modules:
            del sys.modules[mod]
    import events    as EV
    import gfc       as GFC
    import workcenter as WC
    assert len(EV._events) == 0
    return EV, GFC, WC


# ═════════════════════════════════════════════════════════════════════════════
# SLACK DISPATCH KEY PATCH
# ═════════════════════════════════════════════════════════════════════════════

def _patch_slack_dispatch(EV_module) -> None:
    """
    Replace dispatch_key() in the events module with a Slack-based version.

    Slack = due_sim − now − proc_days_cur
    Lower slack → more urgent → sorted first.

    'now' is approximated by arrive_wc_time (the time the job entered the queue),
    which is available on every job dict at dispatch time.
    """
    def _slack_key(job: dict) -> float:
        now_approx = job.get("arrive_wc_time", 0.0)
        slack = job["due_sim"] - now_approx - job.get("proc_days_cur", 0.0)
        return slack

    EV_module.dispatch_key = _slack_key


# ═════════════════════════════════════════════════════════════════════════════
# UPTIME / DOWNTIME PATCH
# ═════════════════════════════════════════════════════════════════════════════

def _patch_uptime(EV_module, uptime: float, rng_seed: int = 42) -> None:
    """
    Inflate processing times to model planned + unplanned downtime.

    Method: availability-based time inflation.
      Effective processing time = nominal / uptime

    This is the standard approach for DES models where downtime is not
    modelled as explicit breakdown events but as a capacity reduction factor.
    E.g. at 90% uptime, a job taking 1.0 day nominally takes 1/0.9 = 1.111 days.

    Applied by wrapping start_op_on_machine to scale proc_days_cur before
    scheduling EV_OP_DONE.
    """
    if uptime >= 1.0:
        return   # no change needed

    _orig_start = EV_module.start_op_on_machine

    def _patched_start(wc, m, job, now):
        # Scale proc time by 1/uptime for limited WCs only
        import config as _c
        if wc.cap_type == "limited":
            job = dict(job)   # shallow copy to avoid mutating original
            job["proc_days_cur"] = job["proc_days_cur"] / uptime
        return _orig_start(wc, m, job, now)

    EV_module.start_op_on_machine = _patched_start


# ═════════════════════════════════════════════════════════════════════════════
# CORE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    label: str,
    df_wo: pd.DataFrame,
    routing_dict: dict,
    dispatch_rule: str = "EDD",
    alert_sustain: float | None = None,
    warn_sustain:  float | None = None,
    uptime: float = 1.0,
) -> dict:

    print(f"\n  ► {label}")

    # ── Override config params for this run ───────────────────────────────────
    orig_dispatch = _cfg.DISPATCH_RULE
    orig_alert    = _cfg.ALERT_SUSTAIN_DAYS
    orig_warn     = _cfg.WARNING_SUSTAIN_DAYS

    _cfg.DISPATCH_RULE        = dispatch_rule
    if alert_sustain is not None:
        _cfg.ALERT_SUSTAIN_DAYS   = alert_sustain
    if warn_sustain is not None:
        _cfg.WARNING_SUSTAIN_DAYS = warn_sustain

    # ── Fresh modules ─────────────────────────────────────────────────────────
    EV, GFC, WC = _fresh()

    # Patch dispatch rule
    if dispatch_rule == "SLACK":
        _patch_slack_dispatch(EV)

    # Patch uptime
    if uptime < 1.0:
        _patch_uptime(EV, uptime)

    # ── Build & run ───────────────────────────────────────────────────────────
    workcenters = WC.build_workcenters()
    EV.schedule_releases(df_wo, routing_dict)
    EV.schedule_gfc_checks()
    EV.run_simulation(workcenters)

    # ── Restore config ────────────────────────────────────────────────────────
    _cfg.DISPATCH_RULE        = orig_dispatch
    _cfg.ALERT_SUSTAIN_DAYS   = orig_alert
    _cfg.WARNING_SUSTAIN_DAYS = orig_warn

    # ── Collect results ───────────────────────────────────────────────────────
    df_done = pd.DataFrame(EV.wo_completed)
    df_ops  = pd.DataFrame(EV.op_records)

    df_done = df_done[df_done["finish_sim"] >= _cfg.WARMUP_DAYS] if len(df_done) else df_done
    df_ops  = df_ops[df_ops["finish_time"]  >= _cfg.WARMUP_DAYS] if len(df_ops)  else df_ops

    n_comp  = len(df_done)
    n_tardy = int(df_done["tardy"].sum()) if n_comp else 0
    pct_ota = 100 * (1 - n_tardy / n_comp) if n_comp else 0.0
    avg_lt  = float(df_done["lead_time"].mean())            if n_comp else 0.0
    p95_lt  = float(df_done["lead_time"].quantile(0.95))    if n_comp else 0.0
    p50_lt  = float(df_done["lead_time"].quantile(0.50))    if n_comp else 0.0
    avg_tdy = float(df_done.loc[df_done["tardy"], "tardiness"].mean()) if n_tardy else 0.0

    # Per-WC util and queue
    util_data: dict[str, dict] = {}
    for wc_name, wc in sorted(workcenters.items()):
        if wc.cap_type == "unlimited" or wc.n_machines == 0:
            continue
        sim_days = _cfg.SIM_WORKING_DAYS if wc.dept == "MANUAL" else _cfg.SIM_END_DAYS
        avail  = wc.n_machines * wc.hours_per_day * sim_days / 24.0
        busy   = sum(m.total_busy for m in wc.machines)
        util   = 100.0 * busy / avail if avail else 0.0
        wc_ops = df_ops[df_ops["wc"] == wc_name] if len(df_ops) else pd.DataFrame()
        mean_q = float(wc_ops["queue_time"].mean()) if len(wc_ops) else 0.0
        max_q  = float(wc_ops["queue_time"].max())  if len(wc_ops) else 0.0
        util_data[wc_name] = {"util": util, "mean_q": mean_q, "max_q": max_q,
                               "n_mach": wc.n_machines}

    n_exp = len(GFC.expansion_log)
    print(f"     OTA={pct_ota:.1f}%  mean_LT={avg_lt:.1f}d  P95={p95_lt:.1f}d  "
          f"tardy={n_tardy}  expansions={n_exp}")

    return {
        "label":       label,
        "n_comp":      n_comp,
        "n_tardy":     n_tardy,
        "pct_ota":     pct_ota,
        "avg_lt":      avg_lt,
        "p50_lt":      p50_lt,
        "p95_lt":      p95_lt,
        "avg_tard":    avg_tdy,
        "util_data":   util_data,
        "df_done":     df_done,
        "df_ops":      df_ops,
        "workcenters": workcenters,
        "expansion_log": GFC.expansion_log,
        "n_expansions": n_exp,
        "uptime":      uptime,
        "dispatch":    dispatch_rule,
        "alert_sustain": alert_sustain or _cfg.ALERT_SUSTAIN_DAYS,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _summary_rows(results: list[dict], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for r in results:
        ud  = r["util_data"]
        bw  = ud.get("WC_BW",       {})
        ahs = ud.get("WC_AMILL_HS", {})
        mm  = ud.get("WC_MMILL",    {})
        row = {
            "Experiment":        r["label"],
            "OTA %":             round(r["pct_ota"],  1),
            "Mean LT (d)":       round(r["avg_lt"],   1),
            "P50 LT (d)":        round(r["p50_lt"],   1),
            "P95 LT (d)":        round(r["p95_lt"],   1),
            "Tardy WOs":         r["n_tardy"],
            "Avg tardiness (d)": round(r["avg_tard"], 1),
            "WC_BW util %":      round(bw.get("util",   0), 1),
            "WC_BW mean Q (d)":  round(bw.get("mean_q", 0), 2),
            "WC_BW max Q (d)":   round(bw.get("max_q",  0), 1),
            "AMILL_HS util %":   round(ahs.get("util",  0), 1),
            "MMILL util %":      round(mm.get("util",   0), 1),
            "GFC expansions":    r["n_expansions"],
        }
        for c in group_cols:
            row[c] = r.get(c, "")
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def _sd(t: float) -> datetime:
    return datetime.combine(_cfg.SIM_ORIGIN + timedelta(days=int(t)),
                             datetime.min.time())

def _fmt_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

CMAP_A = plt.cm.tab10
CMAP_B = plt.cm.Set2
CMAP_C = plt.cm.Reds


def _cdf(series: pd.Series, ax, color, ls, label):
    s = np.sort(series.dropna().values)
    c = np.arange(1, len(s) + 1) / len(s) * 100
    ax.plot(s, c, color=color, linestyle=ls, linewidth=1.5, label=label)


def plot_experiment_group(
    results: list[dict],
    title: str,
    colors,
    out_path: Path,
    contractual_lt: float = 50.0,
) -> None:
    """
    3-panel plot for one experiment group:
      Left   : Lead-time CDF
      Centre : WC_BW queue depth over time
      Right  : Per-WC utilisation bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    ls_cycle = ["-", "--", "-.", ":", (0, (5, 1))]

    for i, (r, ls) in enumerate(zip(results, ls_cycle)):
        col = colors(i / max(len(results) - 1, 1))
        lbl = r["label"]

        # ── CDF ──────────────────────────────────────────────────────────────
        if len(r["df_done"]):
            _cdf(r["df_done"]["lead_time"], axes[0], col, ls,
                 f"{lbl}  (P95={r['p95_lt']:.1f}d)")

        # ── WC_BW queue over time ─────────────────────────────────────────────
        wc_bw = r["workcenters"].get("WC_BW")
        if wc_bw and wc_bw.queue_snaps:
            ts, qs = zip(*wc_bw.queue_snaps)
            qs_s = pd.Series(qs).rolling(14, min_periods=1, center=True).mean()
            axes[1].plot([_sd(t) for t in ts], qs_s.values,
                         color=col, linestyle=ls, linewidth=1.1, alpha=0.9,
                         label=lbl)

    # CDF decorations
    axes[0].axvline(contractual_lt, color="red", linestyle=":", linewidth=1.2,
                    label=f"{contractual_lt:.0f}d contractual LT")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_ylabel("Cumulative %")
    axes[0].set_title("WO Lead-Time CDF")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_xlim(0)
    axes[0].set_ylim(0, 102)

    # Queue decorations
    axes[1].set_title("WC_BW Queue Depth (14-day rolling)")
    axes[1].set_ylabel("Jobs in queue")
    axes[1].legend(fontsize=7)
    _fmt_x(axes[1])
    axes[1].grid(True, alpha=0.25)

    # ── Utilisation bar ───────────────────────────────────────────────────────
    wc_names = sorted({wc for r in results for wc in r["util_data"]})
    n_exp    = len(results)
    bw_bar   = 0.8 / n_exp
    x        = np.arange(len(wc_names))

    for i, r in enumerate(results):
        col = colors(i / max(n_exp - 1, 1))
        utils = [r["util_data"].get(wc, {}).get("util", 0.0) for wc in wc_names]
        axes[2].bar(x + i * bw_bar, utils, width=bw_bar, color=col,
                    alpha=0.8, label=r["label"], edgecolor="white")

    axes[2].axhline(85, color="red",    linestyle="--", linewidth=1,   alpha=0.7, label="85% alert")
    axes[2].axhline(70, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="70% warning")
    axes[2].set_xticks(x + bw_bar * (n_exp - 1) / 2)
    axes[2].set_xticklabels(wc_names, rotation=40, ha="right", fontsize=7)
    axes[2].set_ylabel("Utilisation %")
    axes[2].set_title("Workcenter Utilisation")
    axes[2].set_ylim(0, 110)
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path.name}")


def plot_uptime_detail(results: list[dict], out_path: Path) -> None:
    """
    Extra plot for uptime experiment: shows how utilisation and lead time
    degrade as availability falls, for the two most-loaded WCs.
    """
    uptimes  = [r["uptime"] * 100 for r in results]
    p95_lts  = [r["p95_lt"]       for r in results]
    ota_vals = [r["pct_ota"]       for r in results]
    bw_utils = [r["util_data"].get("WC_BW",       {}).get("util", 0) for r in results]
    ah_utils = [r["util_data"].get("WC_AMILL_HS", {}).get("util", 0) for r in results]
    bw_maxq  = [r["util_data"].get("WC_BW",       {}).get("max_q",0) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("EXP-C: Machine Availability Sensitivity Analysis",
                 fontsize=11, fontweight="bold")

    # P95 LT vs uptime
    axes[0].plot(uptimes, p95_lts, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].axhline(50, color="red", linestyle="--", label="50d contractual")
    for u, v in zip(uptimes, p95_lts):
        axes[0].annotate(f"{v:.1f}d", (u, v), textcoords="offset points",
                          xytext=(5, 5), fontsize=8)
    axes[0].set_xlabel("Machine availability (%)")
    axes[0].set_ylabel("P95 lead time (days)")
    axes[0].set_title("P95 Lead Time vs Availability")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(83, 102)

    # WC utilisation vs uptime
    axes[1].plot(uptimes, bw_utils, "s-", color="steelblue",  linewidth=2, markersize=8, label="WC_BW")
    axes[1].plot(uptimes, ah_utils, "^-", color="darkorange",  linewidth=2, markersize=8, label="WC_AMILL_HS")
    axes[1].axhline(85, color="red",    linestyle="--", linewidth=1, alpha=0.7, label="85% alert")
    axes[1].axhline(70, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="70% warn")
    for u, v in zip(uptimes, bw_utils):
        axes[1].annotate(f"{v:.0f}%", (u, v), textcoords="offset points",
                          xytext=(5, -10), fontsize=7, color="steelblue")
    axes[1].set_xlabel("Machine availability (%)")
    axes[1].set_ylabel("Effective utilisation (%)")
    axes[1].set_title("Effective Utilisation vs Availability")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(83, 102)
    axes[1].set_ylim(0, 110)

    # On-time delivery vs uptime
    axes[2].plot(uptimes, ota_vals, "D-", color="seagreen", linewidth=2, markersize=8)
    axes[2].axhline(95, color="red", linestyle="--", label="95% target")
    for u, v in zip(uptimes, ota_vals):
        axes[2].annotate(f"{v:.1f}%", (u, v), textcoords="offset points",
                          xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel("Machine availability (%)")
    axes[2].set_ylabel("On-time delivery %")
    axes[2].set_title("On-Time Delivery vs Availability")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(83, 102)
    axes[2].set_ylim(80, 102)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {out_path.name}")


def plot_master_summary(
    dispatch_results: list[dict],
    gfc_results:      list[dict],
    uptime_results:   list[dict],
    out_path: Path,
) -> None:
    """
    Single-page summary heatmap: all experiments × key metrics.
    """
    all_results = dispatch_results + gfc_results + uptime_results
    metrics = ["OTA %", "P95 LT (d)", "Mean LT (d)", "Tardy WOs",
               "WC_BW util %", "WC_BW max Q (d)", "AMILL_HS util %", "GFC expansions"]

    df_a = _summary_rows(dispatch_results, [])
    df_b = _summary_rows(gfc_results,      [])
    df_c = _summary_rows(uptime_results,   [])
    df_all = pd.concat([df_a, df_b, df_c], ignore_index=True)

    mat = df_all[metrics].values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(df_all) + 2.5))
    im = ax.imshow(mat.T, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(df_all)))
    ax.set_xticklabels(df_all["Experiment"], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8.5)
    ax.set_title("Sensitivity Analysis — All Experiments × Key Metrics\n"
                 "(green = better / lower risk, red = worse)",
                 fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(df_all)):
        for j, m in enumerate(metrics):
            v = mat[i, j]
            fmt = f"{v:.1f}" if v < 1000 else f"{v:.0f}"
            ax.text(i, j, fmt, ha="center", va="center", fontsize=7,
                    color="black")

    # Group dividers
    ax.axvline(len(dispatch_results) - 0.5, color="white", linewidth=2.5)
    ax.axvline(len(dispatch_results) + len(gfc_results) - 0.5,
               color="white", linewidth=2.5)

    # Group labels above
    def _span_label(ax, x0, x1, text, y=1.03):
        mid = (x0 + x1) / 2
        ax.annotate(text, xy=(mid, y), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="navy")

    _span_label(ax, 0, len(dispatch_results) - 1, "EXP-A: Dispatching rule")
    _span_label(ax, len(dispatch_results),
                len(dispatch_results) + len(gfc_results) - 1,
                "EXP-B: GFC alert sustain")
    _span_label(ax, len(dispatch_results) + len(gfc_results),
                len(df_all) - 1, "EXP-C: Machine uptime")

    plt.colorbar(im, ax=ax, fraction=0.015, label="Relative scale (row-normalised)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Master summary plot → {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*70}")
    print("  SENSITIVITY ANALYSIS — Dispatching / GFC / Machine Uptime")
    print(f"{'#'*70}")

    OUT    = _cfg.HERE / "output" / "sensitivity_analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder : {OUT}")

    from data_loader import load_data
    df_wo_base, routing_dict = load_data(_cfg.DATA_FILE)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: Dispatching rules
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  EXP-A: Dispatching Rule Comparison")
    print(f"{'─'*70}")

    dispatch_results = []
    for exp in DISPATCH_EXPERIMENTS:
        r = run_experiment(
            label        = exp["label"],
            df_wo        = df_wo_base,
            routing_dict = routing_dict,
            dispatch_rule= exp["rule"],
        )
        r["exp_id"] = exp["id"]
        dispatch_results.append(r)

    df_a = _summary_rows(dispatch_results, ["dispatch"])
    df_a.to_csv(OUT / "sensitivity_A_dispatch.csv", index=False)

    plot_experiment_group(
        dispatch_results,
        title     = "EXP-A — Dispatching Rule Comparison (Baseline demand)",
        colors    = CMAP_A,
        out_path  = OUT / "sensitivity_A_dispatch.png",
    )

    print(f"\n  EXP-A Summary:")
    print(df_a[["Experiment","OTA %","P95 LT (d)","Tardy WOs",
                "WC_BW max Q (d)","GFC expansions"]].to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-B: GFC alert sustain threshold
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  EXP-B: GFC Alert-Sustain Threshold Sensitivity")
    print(f"{'─'*70}")

    gfc_results = []
    for exp in GFC_EXPERIMENTS:
        r = run_experiment(
            label         = exp["label"],
            df_wo         = df_wo_base,
            routing_dict  = routing_dict,
            dispatch_rule = "EDD",
            alert_sustain = exp["alert_sustain"],
            warn_sustain  = exp["warn_sustain"],
        )
        r["exp_id"] = exp["id"]
        gfc_results.append(r)

    df_b = _summary_rows(gfc_results, ["alert_sustain"])
    df_b.to_csv(OUT / "sensitivity_B_gfc.csv", index=False)

    plot_experiment_group(
        gfc_results,
        title    = "EXP-B — GFC Alert-Sustain Threshold (Baseline demand)",
        colors   = CMAP_B,
        out_path = OUT / "sensitivity_B_gfc.png",
    )

    print(f"\n  EXP-B Summary:")
    print(df_b[["Experiment","OTA %","P95 LT (d)","Tardy WOs",
                "WC_BW max Q (d)","GFC expansions"]].to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Machine uptime / availability
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  EXP-C: Machine Availability Sensitivity")
    print(f"{'─'*70}")

    uptime_results = []
    for exp in UPTIME_EXPERIMENTS:
        r = run_experiment(
            label        = exp["label"],
            df_wo        = df_wo_base,
            routing_dict = routing_dict,
            dispatch_rule= "EDD",
            uptime       = exp["uptime"],
        )
        r["exp_id"] = exp["id"]
        uptime_results.append(r)

    df_c = _summary_rows(uptime_results, ["uptime"])
    df_c.to_csv(OUT / "sensitivity_C_uptime.csv", index=False)

    plot_experiment_group(
        uptime_results,
        title    = "EXP-C — Machine Availability Sensitivity (Baseline demand)",
        colors   = CMAP_C,
        out_path = OUT / "sensitivity_C_uptime.png",
    )
    plot_uptime_detail(uptime_results, OUT / "sensitivity_C_uptime_detail.png")

    print(f"\n  EXP-C Summary:")
    print(df_c[["Experiment","OTA %","P95 LT (d)","Tardy WOs",
                "WC_BW util %","AMILL_HS util %","GFC expansions"]].to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    plot_master_summary(dispatch_results, gfc_results, uptime_results,
                        OUT / "sensitivity_master.png")

    # Combined CSV
    df_all = pd.concat([df_a, df_b, df_c], ignore_index=True)
    df_all.to_csv(OUT / "sensitivity_all.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PRINTED CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}")

    # Best dispatch rule by P95 LT
    best_d = min(dispatch_results, key=lambda r: r["p95_lt"])
    print(f"\n  EXP-A — Best dispatching rule: {best_d['label']}")
    print(f"    P95 LT = {best_d['p95_lt']:.1f} d  |  OTA = {best_d['pct_ota']:.1f}%  "
          f"|  Tardy = {best_d['n_tardy']}")
    base_d = dispatch_results[0]
    print(f"    vs EDD baseline: ΔP95 = {best_d['p95_lt']-base_d['p95_lt']:+.1f} d  "
          f"  ΔTardy = {best_d['n_tardy']-base_d['n_tardy']:+d}")

    # GFC threshold effect
    print(f"\n  EXP-B — GFC alert threshold effect on WC_BW max queue:")
    for r in gfc_results:
        bw_maxq = r["util_data"].get("WC_BW", {}).get("max_q", 0)
        print(f"    {r['label']:<42}  max_Q = {bw_maxq:.1f} d  "
              f"P95 = {r['p95_lt']:.1f} d  expansions = {r['n_expansions']}")

    # Uptime break-even
    print(f"\n  EXP-C — Uptime impact:")
    contractual = 50.0
    for r in uptime_results:
        flag = " ◄ BREACH" if r["p95_lt"] > contractual else ""
        print(f"    {r['label']:<38}  P95 = {r['p95_lt']:.1f} d  "
              f"OTA = {r['pct_ota']:.1f}%{flag}")

    print(f"\n{'#'*70}")
    print("  All outputs written to ./output/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
