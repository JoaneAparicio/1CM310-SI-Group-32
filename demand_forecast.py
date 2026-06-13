"""
demand_forecast.py — SDG Shopfloor Simulation
===============================================
Step 0: Demand Forecast → Capacity Load Analysis → Proactive Investment Decision

Workflow (mirrors real operations planning logic):
  1. Fit demand growth models to historical VIX unit data (2022–2025)
  2. Project WO shopfloor load for 2026–2027 using seasonal pattern from 2025
  3. Translate projected WO load into capacity load per WC using 2025 load ratios
  4. Identify which WCs will breach utilisation thresholds and when
  5. Output investment triggers with timing and cost — feeding into scenario_analysis.py

Outputs
-------
  output/<run>/forecast_demand_models.png     — VIX unit demand + model fits
  output/<run>/forecast_shopfloor_load.png    — projected monthly load per WC
  output/<run>/forecast_capacity_headroom.png — utilisation forecast + breach dates
  output/<run>/forecast_investment_plan.png   — investment decision timeline
  output/<run>/forecast_summary.csv           — numerical summary

Usage
-----
  python demand_forecast.py
"""

from __future__ import annotations
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
import config as _cfg

OUT = HERE / "output" / f"forecast_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
OUT.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUT}")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "legend.fontsize": 8, "legend.framealpha": 0.85, "figure.dpi": 150,
})

WC_COLORS = {
    "WC_BW":       "#c53030",
    "WC_AMILL_HS": "#d97706",
    "WC_MMILL":    "#2b6cb0",
    "WC_AMILL":    "#276749",
    "WC_LATHE":    "#6b21a8",
    "WC_CMM":      "#718096",
    "WC_QC":       "#b7791f",
    "WC_3DP":      "#2c7a7b",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. HISTORICAL DATA
# ══════════════════════════════════════════════════════════════════════════════

# VIX units (from conceptual design Section 4.1)
# 2025 figure: 38 units in Q1–Q3, annualised = 38/0.70 ≈ 56 units
VIX_HIST = {2022: 7, 2023: 15, 2024: 34, 2025: 56}

# Quarterly distribution from conceptual design (22-23-25-30%)
Q_WEIGHTS = [0.22, 0.23, 0.25, 0.30]

# WC annual capacity (days/year) from config
WC_CONFIG = _cfg.WC_CONFIG
def _working_days_in_year(year: int) -> float:
    """Count Mon–Fri days in a given calendar year."""
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    count = 0.0
    while d <= end:
        if d.weekday() < 5:
            count += 1.0
        d += timedelta(days=1)
    return count

WC_CAPACITY = {}
# Use the reference year from the simulation start for annual capacity.
_REF_YEAR = date.fromisoformat(_cfg.SIM_START_STR).year
_WORKING_DAYS_REF = _working_days_in_year(_REF_YEAR)
_CALENDAR_DAYS_REF = 366.0 if (_REF_YEAR % 4 == 0 and
                                (_REF_YEAR % 100 != 0 or _REF_YEAR % 400 == 0)) else 365.0

for wc, (nm, dept, hpd, cap) in WC_CONFIG.items():
    if cap == "limited" and nm:
        # MANUAL WCs are staffed Mon–Fri only → actual working days in ref year.
        # AUTO WCs can run through weekends → calendar days in ref year.
        days = _WORKING_DAYS_REF if dept == "MANUAL" else _CALENDAR_DAYS_REF
        WC_CAPACITY[wc] = nm * hpd * days / 24  # machine-days/year

# 2025 actual load per WC (from S0 simulation results)
# These are the reference ratios we use to project future load
WC_LOAD_2025 = {
    "WC_AMILL":    2698.3,
    "WC_AMILL_HS": 588.2,
    "WC_BW":       875.7,
    "WC_CMM":      487.2,
    "WC_CONV":     102.4,
    "WC_LATHE":    607.7,
    "WC_MMILL":    2567.7,
    "WC_QC":       353.1,
    "WC_3DP":      172.5,
    "WC_SAW":      36.5,
    "WC_REWORK":   0.6,
}

# 2025 monthly WO releases (from data.xlsx MRP)
MONTHLY_WO_2025 = {
    1: 584, 2: 488, 3: 491, 4: 466, 5: 438, 6: 377,
    7: 518, 8: 418, 9: 579, 10: 617, 11: 552, 12: 532
}
ANNUAL_WO_2025 = sum(MONTHLY_WO_2025.values())  # 6060


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEMAND GROWTH MODELS
# ══════════════════════════════════════════════════════════════════════════════

years_hist = np.array(list(VIX_HIST.keys()), dtype=float)
units_hist = np.array(list(VIX_HIST.values()), dtype=float)
years_proj = np.arange(2022, 2029, dtype=float)
years_fut  = np.array([2026, 2027, 2028], dtype=float)


def _linear(x, a, b):
    return a * x + b

def _exponential(x, a, b):
    return a * np.exp(b * (x - 2022))

def _logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def fit_model(func, x, y, p0, bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)
    y_pred  = func(x, *popt)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return popt, pcov, r2


# Fit all three models
p_lin, cov_lin, r2_lin = fit_model(_linear,      years_hist, units_hist, [10, -20000])
p_exp, cov_exp, r2_exp = fit_model(_exponential, years_hist, units_hist, [7, 0.5],
                                    bounds=([0.01, 0.01], [1000, 5]))
p_log, cov_log, r2_log = fit_model(_logistic,    years_hist, units_hist, [200, 1, 2025],
                                    bounds=([50, 0.01, 2020], [1000, 5, 2035]))

models = {
    "Linear":      (_linear,      p_lin, cov_lin, r2_lin,  "#718096", "--"),
    "Exponential": (_exponential, p_exp, cov_exp, r2_exp,  "#c53030", "-"),
    "Logistic":    (_logistic,    p_log, cov_log, r2_log,  "#276749", "-."),
}

# Project each model
projections = {}
for name, (func, popt, pcov, r2, col, ls) in models.items():
    proj = {int(y): round(func(y, *popt), 1) for y in years_proj}
    projections[name] = proj

print("\n=== DEMAND GROWTH MODEL FITS ===")
print(f"{'Model':<14} {'R²':>6}  " + "  ".join(f"{int(y)}" for y in years_proj))
for name, (func, popt, pcov, r2, col, ls) in models.items():
    vals = "  ".join(f"{projections[name][int(y)]:>5.0f}" for y in years_proj)
    print(f"  {name:<12} {r2:>6.3f}  {vals}")

# Scenario alignment
print("\n  Conceptual design scenarios:")
print(f"  S1 Conservative: 2026=72, 2027=88")
print(f"  S2 Base/High:    2026=92, 2027=152")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROJECT SHOPFLOOR WO LOAD
# ══════════════════════════════════════════════════════════════════════════════

def project_wo_load(annual_units_by_year: dict[int, float]) -> pd.DataFrame:
    """
    Project monthly WO releases for given VIX unit forecast.

    Method:
    - Scale 2025 WO total proportionally to projected unit growth
    - Apply 2025 seasonal pattern (monthly indices) to distribute within year
    - Returns DataFrame with columns: year, month, ym, projected_wo, growth_factor
    """
    # 2025 seasonal indices
    seasonal = {m: n / ANNUAL_WO_2025 for m, n in MONTHLY_WO_2025.items()}

    rows = []
    for year, units in annual_units_by_year.items():
        growth = units / VIX_HIST[2025]   # relative to 2025 baseline
        ann_wo = ANNUAL_WO_2025 * growth
        for month in range(1, 13):
            monthly_wo = ann_wo * seasonal[month]
            rows.append({
                "year":          year,
                "month":         month,
                "ym":            f"{year}-{month:02d}",
                "units_annual":  units,
                "growth_factor": growth,
                "projected_wo":  monthly_wo,
            })
    return pd.DataFrame(rows)


def project_wc_load(df_wo: pd.DataFrame) -> pd.DataFrame:
    """
    Project monthly capacity load per WC from projected WO volumes.

    Method: load per WC scales linearly with WO volume (constant mix assumption).
    Reference: 2025 actual load per WC per WO (from simulation results).
    """
    # Load per WO per WC (2025 baseline)
    load_per_wo = {wc: WC_LOAD_2025.get(wc, 0) / ANNUAL_WO_2025
                   for wc in WC_LOAD_2025}

    rows = []
    for _, r in df_wo.iterrows():
        for wc, lpwo in load_per_wo.items():
            load = r["projected_wo"] * lpwo
            cap  = WC_CAPACITY.get(wc, 1)
            # Monthly capacity = annual / 12
            cap_mo = cap / 12
            rows.append({
                "ym":             r["ym"],
                "year":           r["year"],
                "month":          r["month"],
                "wc":             wc,
                "projected_load": load,
                "capacity_mo":    cap_mo,
                "util_pct":       100 * load / cap_mo if cap_mo > 0 else 0,
                "growth_factor":  r["growth_factor"],
            })
    return pd.DataFrame(rows)


# Build projections for each model + scenario bounds
scenario_units = {
    "Exponential": {y: projections["Exponential"][y] for y in range(2025, 2029)},
    "Linear":      {y: projections["Linear"][y]      for y in range(2025, 2029)},
    "Logistic":    {y: projections["Logistic"][y]    for y in range(2025, 2029)},
    "S1_Conservative": {2025: 56, 2026: 72,  2027: 88,  2028: 105},
    "S2_Base_High":    {2025: 56, 2026: 92,  2027: 152, 2028: 200},
}

wc_load_by_scenario = {}
for sc_name, units_by_year in scenario_units.items():
    df_wo   = project_wo_load(units_by_year)
    df_load = project_wc_load(df_wo)
    wc_load_by_scenario[sc_name] = df_load

# Key output: annual utilisation per WC per scenario
print("\n=== PROJECTED WC UTILISATION (annual average) ===")
focus_wcs = ["WC_BW", "WC_AMILL_HS", "WC_MMILL", "WC_AMILL", "WC_LATHE"]
for sc_name in ["Exponential", "S1_Conservative", "S2_Base_High"]:
    df = wc_load_by_scenario[sc_name]
    ann = df.groupby(["year","wc"])["util_pct"].mean().reset_index()
    print(f"\n  {sc_name}:")
    print(f"  {'WC':<14}", end="")
    for yr in [2025, 2026, 2027, 2028]:
        print(f"  {yr}", end="")
    print()
    for wc in focus_wcs:
        print(f"  {wc:<14}", end="")
        for yr in [2025, 2026, 2027, 2028]:
            v = ann[(ann.year==yr) & (ann.wc==wc)]['util_pct']
            val = f"{v.values[0]:.0f}%" if len(v) else "–"
            print(f"  {val:>5}", end="")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. BREACH DETECTION & INVESTMENT TRIGGERS
# ══════════════════════════════════════════════════════════════════════════════

UTIL_ALERT   = _cfg.UTIL_ALERT    # 85%
UTIL_WARNING = _cfg.UTIL_WARNING  # 70%
MACHINE_LEAD = _cfg.MACHINE_LEAD_DAYS  # 180d

def detect_breaches(df_load: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Find first month each WC breaches warning/alert thresholds."""
    rows = []
    for wc in df_load["wc"].unique():
        sub = df_load[df_load["wc"] == wc].sort_values("ym")
        # Sustained breach: 2 consecutive months above threshold
        for thresh, label in [(UTIL_ALERT, "ALERT"), (UTIL_WARNING, "WARNING")]:
            above = sub[sub["util_pct"] > thresh]
            if len(above) >= 1:
                # Find first sustained breach (2 consecutive months)
                for i in range(len(above) - 1):
                    ym1 = above.iloc[i]["ym"]
                    ym2 = above.iloc[i+1]["ym"]
                    y1, m1 = int(ym1[:4]), int(ym1[5:])
                    y2, m2 = int(ym2[:4]), int(ym2[5:])
                    if (y2 - y1) * 12 + (m2 - m1) == 1:
                        util_at_breach = above.iloc[i]["util_pct"]
                        breach_date = date(y1, m1, 1)
                        # Investment should be ordered MACHINE_LEAD days before
                        order_date = breach_date - timedelta(days=MACHINE_LEAD)
                        rows.append({
                            "scenario":    scenario,
                            "wc":          wc,
                            "threshold":   label,
                            "util_pct":    round(util_at_breach, 1),
                            "breach_ym":   ym1,
                            "breach_date": breach_date,
                            "order_date":  order_date,
                            "already_late": order_date < date.today(),
                        })
                        break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


breaches_all = {}
for sc_name in ["Exponential", "S1_Conservative", "S2_Base_High"]:
    df = wc_load_by_scenario[sc_name]
    br = detect_breaches(df, sc_name)
    breaches_all[sc_name] = br
    if len(br):
        print(f"\n=== BREACH DETECTION: {sc_name} ===")
        print(br[["wc","threshold","util_pct","breach_ym","order_date","already_late"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

def _ym_to_dt(ym: str) -> datetime:
    y, m = int(ym[:4]), int(ym[5:])
    return datetime(y, m, 15)


# ── Fig F1: VIX demand growth models ──────────────────────────────────────────
def plot_demand_models():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Fig F1 — VIX Demand Growth: Historical Data & Model Projections", fontsize=11)

    # Left: model fits with CI
    ax1.scatter(years_hist, units_hist, s=90, zorder=10,
                color="#1a202c", label="Historical (conceptual design)", clip_on=False)

    for name, (func, popt, pcov, r2, col, ls) in models.items():
        y_fit = func(years_proj, *popt)
        ax1.plot(years_proj, y_fit, color=col, linestyle=ls, linewidth=2,
                 label=f"{name}  (R²={r2:.3f})")

        # Confidence interval via MC sampling (500 draws from parameter distribution)
        try:
            n_draws = 500
            param_draws = np.random.multivariate_normal(popt, pcov, size=n_draws)
            y_draws = np.array([func(years_proj, *p) for p in param_draws
                                if np.all(np.isfinite(func(years_proj, *p)))])
            if len(y_draws) > 10:
                y_lo = np.percentile(y_draws, 10, axis=0)
                y_hi = np.percentile(y_draws, 90, axis=0)
                ax1.fill_between(years_proj, y_lo, y_hi, color=col, alpha=0.08)
        except Exception:
            pass

    # Scenario markers
    ax1.axhline(72,  color="#d97706", linestyle=":", linewidth=1.2, alpha=0.7, label="S1 Conserv. 2026 (72)")
    ax1.axhline(92,  color="#c53030", linestyle=":", linewidth=1.2, alpha=0.7, label="S2 Base 2026 (92)")
    ax1.axhline(152, color="#9b2c2c", linestyle=":", linewidth=1.2, alpha=0.5, label="S2 Base 2027 (152)")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("VIX units / year")
    ax1.set_title("Demand Growth Models (4 historical data points)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.set_xlim(2021.5, 2028.5)
    ax1.set_ylim(0, 220)
    ax1.set_xticks(range(2022, 2029))

    # Right: model comparison table + growth rates
    ax2.axis("off")
    years_show = [2025, 2026, 2027, 2028]
    tbl_data = []
    tbl_data.append(["Historical", "7", "15", "34", "56", "–", "–"])
    for name, (func, popt, pcov, r2, col, ls) in models.items():
        row = [name]
        for yr in years_show:
            v = projections[name][yr]
            row.append(f"{v:.0f}")
        # YoY growth rates 2025→2026 and 2026→2027
        v26 = projections[name][2026]
        v27 = projections[name][2027]
        row.append(f"{100*(v26/56-1):.0f}%")
        row.append(f"{100*(v27/v26-1):.0f}%")
        tbl_data.append(row)

    # Add scenarios
    for sc_name, units_yr in [("S1 Conservative", {2026:72, 2027:88, 2028:105}),
                               ("S2 Base/High",    {2026:92, 2027:152, 2028:200})]:
        row = [sc_name, "56"]
        for yr in years_show[1:]:
            row.append(str(units_yr.get(yr, "–")))
        row.append(f"{100*(units_yr[2026]/56-1):.0f}%")
        row.append(f"{100*(units_yr[2027]/units_yr[2026]-1):.0f}%")
        tbl_data.append(row)

    col_labels = ["Model", "2025", "2026", "2027", "2028", "YoY 25→26", "YoY 26→27"]
    tbl = ax2.table(
        cellText  = tbl_data,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0.15, 1, 0.78],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#f7fafc")  # historical row
        elif r in [4, 5]:
            cell.set_facecolor("#fffbeb")  # scenario rows

    ax2.text(0.5, 0.06,
             "Exponential model best fits the observed acceleration.\n"
             "Logistic model assumes market saturation around 150–200 units.\n"
             "S1/S2 scenarios bracket the plausible range for planning.",
             ha="center", va="bottom", transform=ax2.transAxes,
             fontsize=8, style="italic", color="#4a5568",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ebf8ff",
                       edgecolor="#2b6cb0", alpha=0.9))
    ax2.set_title("Projection Comparison & Annual Growth Rates", fontsize=9)

    plt.tight_layout()
    p = OUT / "fig_F1_demand_models.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")
    return p


# ── Fig F2: Projected shopfloor utilisation over time ─────────────────────────
def plot_shopfloor_load():
    focus_wcs = ["WC_BW", "WC_AMILL_HS", "WC_MMILL", "WC_AMILL", "WC_LATHE", "WC_CMM"]
    n_wcs = len(focus_wcs)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    fig.suptitle("Fig F2 — Projected Monthly Workcenter Utilisation by Demand Scenario\n"
                 "(Based on 2025 load ratios scaled by WO volume growth)", fontsize=10)
    axes_flat = axes.flatten()

    sc_styles = {
        "Exponential":    ("#c53030", "-",  "Exponential model"),
        "Linear":         ("#718096", "--", "Linear model"),
        "Logistic":       ("#276749", "-.", "Logistic model"),
        "S1_Conservative":("#d97706", ":",  "S1 Conservative"),
        "S2_Base_High":   ("#9b2c2c", (0,(5,1)), "S2 Base/High"),
    }

    for ax, wc in zip(axes_flat, focus_wcs):
        for sc_name, (col, ls, label) in sc_styles.items():
            df = wc_load_by_scenario[sc_name]
            sub = df[df["wc"] == wc].sort_values("ym")
            if sub.empty:
                continue
            dts = [_ym_to_dt(ym) for ym in sub["ym"]]
            # Smooth: 3-month rolling average
            util_s = sub["util_pct"].rolling(3, min_periods=1, center=True).mean()
            ax.plot(dts, util_s.values, color=col, linestyle=ls,
                    linewidth=1.4 if "Conservative" not in label and "Base" not in label else 1.8,
                    alpha=0.85, label=label)

        ax.axhline(85, color="#c53030", linestyle="--", linewidth=1, alpha=0.7, label="85% alert")
        ax.axhline(70, color="#d97706", linestyle=":",  linewidth=0.9, alpha=0.6, label="70% warning")
        ax.axhline(100, color="#1a202c", linestyle="-", linewidth=0.7, alpha=0.4)

        # Shade the capacity breach zone
        ax.fill_between([datetime(2025,1,1), datetime(2029,1,1)],
                         85, 110, alpha=0.04, color="#c53030")

        ax.set_title(wc, fontweight="bold")
        ax.set_ylabel("Utilisation %")
        ax.set_ylim(0, 115)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.2)

        # Add 2025 actual as reference point
        act_util = 100 * WC_LOAD_2025.get(wc, 0) / (WC_CAPACITY.get(wc, 1))
        ax.scatter([datetime(2025, 7, 1)], [act_util], color="#1a202c",
                   s=60, zorder=10, marker="D", label=f"2025 actual ({act_util:.0f}%)")

        if wc == "WC_BW":
            ax.legend(fontsize=6.5, loc="upper left", ncol=2)
        else:
            ax.legend(fontsize=6.5, loc="upper left")

    plt.tight_layout()
    p = OUT / "fig_F2_shopfloor_load.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")
    return p


# ── Fig F3: Capacity headroom & breach dates ──────────────────────────────────
def plot_capacity_headroom():
    focus_wcs = ["WC_BW", "WC_AMILL_HS", "WC_MMILL", "WC_LATHE", "WC_AMILL", "WC_CMM"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Fig F3 — Capacity Headroom Analysis: When Does Each WC Breach?\n"
                 "(Annual average utilisation — shaded area = capacity risk zone)",
                 fontsize=11)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    sc_plot = {
        "S1_Conservative": ("#d97706", "--", "S1 Conservative"),
        "Exponential":     ("#c05621", "-",  "Exponential forecast"),
        "S2_Base_High":    ("#c53030", "-.", "S2 Base/High"),
    }

    years_axis = [2025, 2026, 2027, 2028]

    for idx, wc in enumerate(focus_wcs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        cap_ann = WC_CAPACITY.get(wc, 1)

        for sc_name, (col, ls, label) in sc_plot.items():
            df = wc_load_by_scenario[sc_name]
            ann = df[df["wc"] == wc].groupby("year")["util_pct"].mean().reset_index()
            ann = ann[ann["year"].isin(years_axis)]
            ax.plot(ann["year"], ann["util_pct"], color=col, linestyle=ls,
                    linewidth=2, marker="o", markersize=7, label=label)

            # Annotate each year
            for _, row in ann.iterrows():
                ax.annotate(f"{row['util_pct']:.0f}%",
                            (row["year"], row["util_pct"]),
                            textcoords="offset points", xytext=(4, 5),
                            fontsize=7, color=col)

        # Actual 2025
        act_util = 100 * WC_LOAD_2025.get(wc, 0) / cap_ann
        ax.scatter([2025], [act_util], color="#1a202c", s=80, zorder=10,
                   marker="*", label=f"2025 actual ({act_util:.0f}%)")

        ax.axhline(85,  color="#c53030", linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(70,  color="#d97706", linestyle=":",  linewidth=0.8, alpha=0.7)
        ax.axhline(100, color="#1a202c", linestyle="-",  linewidth=0.7, alpha=0.3)
        ax.fill_between(years_axis, 85, 105, alpha=0.06, color="#c53030")
        ax.fill_between(years_axis, 70, 85,  alpha=0.04, color="#d97706")

        ax.set_title(wc, fontweight="bold")
        ax.set_ylabel("Annual avg. utilisation %")
        ax.set_ylim(0, 115)
        ax.set_xticks(years_axis)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = OUT / "fig_F3_capacity_headroom.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")
    return p


# ── Fig F4: Investment decision timeline ──────────────────────────────────────
def plot_investment_timeline():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(15, 9),
                                          gridspec_kw={"height_ratios": [1.4, 1]})
    fig.suptitle("Fig F4 — Forecast-Driven Investment Plan: WC_BW Proactive Decision Timeline",
                 fontsize=11)

    # Top: WC_BW utilisation projection with investment events
    sc_plot = {
        "S1_Conservative": ("#d97706", "--", "S1 Conservative"),
        "Exponential":     ("#c05621", "-",  "Exponential forecast"),
        "S2_Base_High":    ("#c53030", "-.", "S2 Base/High"),
    }

    for sc_name, (col, ls, label) in sc_plot.items():
        df = wc_load_by_scenario[sc_name]
        sub = df[df["wc"] == "WC_BW"].sort_values("ym")
        dts = [_ym_to_dt(ym) for ym in sub["ym"]]
        util_s = sub["util_pct"].rolling(3, min_periods=1, center=True).mean()
        ax_top.plot(dts, util_s.values, color=col, linestyle=ls,
                    linewidth=2, alpha=0.9, label=label)

    ax_top.axhline(85,  color="#c53030", linestyle="--", linewidth=1.2, alpha=0.8,
                   label="85% alert threshold")
    ax_top.axhline(70,  color="#d97706", linestyle=":",  linewidth=1, alpha=0.7,
                   label="70% warning threshold")
    ax_top.axhline(100, color="#1a202c", linestyle="-",  linewidth=0.7, alpha=0.4,
                   label="100% capacity ceiling")
    ax_top.fill_between([datetime(2025,1,1), datetime(2029,1,1)],
                         85, 115, alpha=0.05, color="#c53030")

    # Mark key events on the timeline
    events = [
        (datetime(2025, 1, 15),  62,   "⬛ Simulation\nstart",         "#1a202c", "bottom"),
        (datetime(2025, 2, 15),  90,   "🔴 WC_BW already\nat 90%\n(2025 actual)", "#c53030", "top"),
        (datetime(2025, 2, 1),   50,   "✅ ORDER machine\n(day 1)",     "#276749", "bottom"),
        (datetime(2025, 8, 1),   78,   "✅ Machine\nonline (+6 mo)",     "#276749", "bottom"),
    ]
    for dt, y, txt, col, va in events:
        ax_top.annotate(txt, (dt, y),
                        textcoords="offset points",
                        xytext=(0, 15 if va == "top" else -45),
                        ha="center", fontsize=7.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor=col, alpha=0.9))

    # Shade machine lead time
    ax_top.axvspan(datetime(2025,2,1), datetime(2025,8,1),
                   alpha=0.08, color="#276749",
                   label="6-month machine lead time")

    ax_top.set_title("WC_BW Projected Utilisation — With Proactive Investment Logic", fontsize=9)
    ax_top.set_ylabel("Utilisation %")
    ax_top.set_ylim(0, 130)
    ax_top.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_top.grid(True, alpha=0.2)

    # Bottom: decision logic table
    ax_bot.axis("off")
    tbl_data = [
        ["WC_BW",       "90%",  "Jan 2025",  "Already breaching",  "Day 1 of simulation", "Jul 2025",  "€500k + €100k/yr", "✅ Proactive"],
        ["WC_AMILL_HS", "61%",  "Oct 2025",  "S2: Q4 2026 → 85%", "Q1 2026 latest",      "Q3 2026",   "€500k + €100k/yr", "⚠ Monitor"],
        ["WC_MMILL",    "56%",  "–",         "S2: ~70% by 2027",   "Q3 2026",             "Q1 2027",   "Tier-1 first",     "📊 Warning"],
        ["WC_AMILL",    "35%",  "–",         "No breach to 2028",  "–",                   "–",         "–",                "✓ Safe"],
        ["WC_LATHE",    "36%",  "–",         "No breach to 2028",  "–",                   "–",         "–",                "✓ Safe"],
    ]
    col_labels = ["WC", "2025 util", "First breach",
                  "Forecast risk", "Order by", "Online by", "Cost", "Action"]
    tbl = ax_bot.table(
        cellText  = tbl_data,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0.05, 1, 0.88],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if r == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold")
        elif r == 1 and c == 7:
            cell.set_facecolor("#c6f6d5")
            cell.set_text_props(fontweight="bold")
        elif r == 2 and c == 7:
            cell.set_facecolor("#fefcbf")
        elif r == 3 and c == 7:
            cell.set_facecolor("#fefcbf")
        elif r > 3 and c == 7:
            cell.set_facecolor("#c6f6d5")
    ax_bot.set_title("Forecast-Driven Investment Decision Table", fontsize=9, pad=8)

    plt.tight_layout()
    p = OUT / "fig_F4_investment_timeline.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_summary():
    rows = []
    for sc_name in ["Exponential", "Linear", "Logistic", "S1_Conservative", "S2_Base_High"]:
        df = wc_load_by_scenario[sc_name]
        ann = df.groupby(["year","wc"])["util_pct"].mean().reset_index()
        for wc in WC_LOAD_2025.keys():
            for yr in [2025, 2026, 2027, 2028]:
                v = ann[(ann.year==yr) & (ann.wc==wc)]["util_pct"]
                rows.append({
                    "scenario": sc_name,
                    "wc": wc,
                    "year": yr,
                    "util_pct_annual_avg": round(v.values[0], 1) if len(v) else None,
                })
    df_sum = pd.DataFrame(rows)
    p = OUT / "forecast_summary.csv"
    df_sum.to_csv(p, index=False)
    print(f"  Saved: {p.name}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*65}")
    print("  DEMAND FORECAST & CAPACITY LOAD ANALYSIS")
    print(f"{'#'*65}\n")

    print("Generating figures...")
    p1 = plot_demand_models()
    p2 = plot_shopfloor_load()
    p3 = plot_capacity_headroom()
    p4 = plot_investment_timeline()
    p5 = save_summary()

    print(f"\n{'─'*65}")
    print("  KEY FINDINGS:")
    print(f"  • WC_BW is already at 90% utilisation in 2025 with current demand")
    print(f"  • Under ALL demand scenarios, WC_BW breaches 85% by 2026")
    print(f"  • Investment decision should be made NOW (day 1) for machine")
    print(f"    to be online before Q4 2025 peak (6-month lead time)")
    print(f"  • WC_AMILL_HS approaches 85% under S2 in Q4 2026 —")
    print(f"    second Auriga decision needed by Q1 2026 at the latest")
    print(f"  • All other WCs have sufficient headroom through 2027")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
