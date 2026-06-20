"""
config.py — SDG Shopfloor Simulation
=====================================
Centralised configuration. Edit only this file to change simulation parameters.
"""

from datetime import date, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent.resolve()
DATA_FILE = HERE / "data.xlsx"          # override via CLI arg in main.py
OUT       = HERE / "output"

# ── Simulation horizon ────────────────────────────────────────────────────────
WARMUP_DAYS = 90.0   # first 90 days are warmup, not counted for metrics or GFC
SIM_START_STR = "2025-01-01"
SIM_END_STR   = "2026-12-31"

SIM_ORIGIN   = date.fromisoformat(SIM_START_STR)
SIM_END_DAYS = float((date.fromisoformat(SIM_END_STR) - SIM_ORIGIN).days)

# Working days in the simulation horizon (Mon–Fri only).
# Computed exactly from the actual calendar, inclusive of both endpoints.
# Used as denominator for MANUAL workcenter utilisation calculations.
_SIM_END = date.fromisoformat(SIM_END_STR)
_d = SIM_ORIGIN
SIM_WORKING_DAYS = 0.0
while _d <= _SIM_END:
    if _d.weekday() < 5:   # 0=Mon … 4=Fri
        SIM_WORKING_DAYS += 1.0
    _d += timedelta(days=1)

# ── Dispatching ───────────────────────────────────────────────────────────────
DISPATCH_RULE = "EDD"   # "EDD" | "FIFO" | "SPT"

# ── GFC / monitoring parameters ───────────────────────────────────────────────
MONITOR_INTERVAL    = 14.0   # days between GFC checks
ROLLING_WINDOW_DAYS = 60.0   # rolling-average window for utilisation

# Utilisation thresholds
UTIL_WARNING        = 70.0   # rolling-avg % → warning zone  → +2 h/day (Tier-1a)
UTIL_ALERT          = 85.0   # rolling-avg % → alert zone    → jump to 21 h/day (Tier-1b) or +1 machine if already at 21h

GFC_EXPANSIONS_ENABLED = True   # False = only monitor, do not expand

# Sustained-overload timers (days above threshold before action fires)
WARNING_SUSTAIN_DAYS = 14.0  # days above UTIL_WARNING before +2 h step
ALERT_SUSTAIN_DAYS   = 7.0   # days above UTIL_ALERT  before jump-to-21h / machine

EXPAND_NOTICE_DAYS  = 30.0   # Tier-1: hours-extension advance notice (days)
MACHINE_LEAD_DAYS   = 180.0  # Tier-2: lead time from decision to new machine online

# ── Snapshot interval (for utilisation & queue time-series) ──────────────────
SNAP_INTERVAL = 1.0   # days between snapshots (1 = daily)

# ── Transport between WC ──────────────────────────────────────────────────────
TRANSPORT_DELAY_WD = 30 / (16 * 60)   # 30 min expressed in MANUAL working days ≈ 0.031

# ── Workcenter definitions ────────────────────────────────────────────────────
# {wc_name: (n_machines, dept, hours_per_day, capacity_type)}
#   dept        : "AUTO" (21 h/day) | "MANUAL" (16 h/day)
#   capacity_type: "limited" | "unlimited"
WC_CONFIG = {
    "WC_AMILL":    (24,   "AUTO",   21, "limited"),
    "WC_AMILL_HS": ( 3,   "AUTO",   21, "limited"),
    "WC_MMILL":    (19,   "MANUAL", 16, "limited"),
    "WC_LATHE":    ( 7,   "MANUAL", 16, "limited"),
    "WC_BW":       ( 4,   "MANUAL", 16, "limited"),
    "WC_QC":       ( 9,   "MANUAL", 16, "limited"),
    "WC_CMM":      ( 6,   "MANUAL", 16, "limited"),
    "WC_3DP":      ( 6,   "MANUAL", 16, "limited"),
    "WC_SAW":      ( 3,   "MANUAL", 16, "limited"),
    "WC_CONV":     ( 4,   "MANUAL", 16, "limited"),
    "WC_REWORK":   ( 1,   "MANUAL", 16, "limited"),
    "WC_MISC":     (None, "MANUAL", 16, "unlimited"),
    "WC_OP":       (None, "MANUAL", 16, "unlimited"),
}

# Fixed lead times (working days) for unlimited-capacity workcenters
FIXED_LT = {
    "WC_MISC": 2.0,
    "WC_OP":   5.0,
}
