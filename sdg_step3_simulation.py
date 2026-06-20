"""
SDG Shopfloor Simulation — Step 3: GFC Layer Integration
=========================================================
Extends Step 2 (full multi-workcenter routing) with a Goods Flow Control (GFC)
layer that monitors shopfloor performance and triggers capacity expansion decisions.

WHAT IS NEW vs STEP 2:
  - GFC module: periodic utilisation monitoring (every 30 sim-days).
  - Rolling-average utilisation tracked per workcenter (30-day window).
  - Capacity alert fired when rolling utilisation > UTIL_ALERT (85%).
  - Capacity expansion triggered when utilisation remains above UTIL_EXPAND (90%)
    for EXPAND_SUSTAIN_DAYS (60 days) continuously.
  - Two-tier expansion logic:
      Tier 1 — MANUAL WCs: extend operating hours from 16h to 21h (in 1h steps),
               1 month advance notice, costs €1,500/machine/extra-hour/month.
               Effective after EXPAND_NOTICE_DAYS (30) sim-days.
      Tier 2 — Add one machine (any WC), 6-month lead time, €500k CAPEX + €100k/yr OPEX.
               AUTO WCs skip Tier-1 and go straight to Tier-2.
  - Capacity expansion decisions are irreversible; at most one extra machine per WC.
  - Expansion log captures every decision with date and cost estimate.
  - GFC summary report printed at end; expansion events overlaid on utilisation plots.

DESIGN PARAMETERS (edit here):
  DISPATCH_RULE       — EDD | FIFO | SPT
  UTIL_ALERT          — rolling-average threshold that logs a warning   (85%)
  UTIL_EXPAND         — threshold that starts the sustained-overload timer (90%)
  EXPAND_SUSTAIN_DAYS — days above UTIL_EXPAND before expansion triggers  (60)
  MONITOR_INTERVAL    — GFC check cadence in sim-days                     (30)
  EXPAND_NOTICE_DAYS  — advance notice for hours-extension (Tier-1)       (30)
  MACHINE_LEAD_DAYS   — lead time from decision to new machine online      (180)

ASSUMPTIONS BEING VALIDATED (connection to conceptual design):
  B1. At-or-near-85% utilisation on WC_AMILL_HS will trigger a capacity alert early.
  B2. EDD dispatching improves on-time delivery vs FIFO/SPT.
  B3. Tier-1 (hours extension) is sufficient for MANUAL WCs; Tier-2 only needed
      for sustained high-load or AUTO WCs.
  B4. GFC monitoring with a 30-day rolling window detects emerging bottlenecks
      with enough lead time for the 1-month hours-extension notice.

HOW TO RUN:
  Place script in the same folder as data.xlsx:
      python sdg_step3_simulation.py
  Or pass the xlsx path:
      python sdg_step3_simulation.py "C:/path/to/data.xlsx"
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS & PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
import heapq
import sys
import os
from datetime import date, timedelta
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent.resolve()

if len(sys.argv) > 1:
    DATA_FILE = Path(sys.argv[1])
else:
    DATA_FILE = HERE / "data.xlsx"

OUT = HERE / "output" / "step3"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.is_file():
    sys.exit(f"\n[ERROR] File not found: {DATA_FILE}\n"
             "  Pass the full path as argument or place data.xlsx next to the script.\n")

print(f"Data file : {DATA_FILE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SIM_START_STR  = "2025-01-01"
SIM_END_STR    = "2026-12-31"
DISPATCH_RULE  = "EDD"           # "EDD" | "FIFO" | "SPT"

# ── GFC / monitoring parameters ───────────────────────────────────────────────
MONITOR_INTERVAL    = 30.0   # days between GFC checks
UTIL_ALERT          = 85.0   # rolling-avg % → log alert
UTIL_EXPAND         = 90.0   # rolling-avg % → start sustained-timer
EXPAND_SUSTAIN_DAYS = 30.0   # consecutive days above UTIL_EXPAND before expansion
EXPAND_NOTICE_DAYS  = 30.0   # Tier-1: hours-extension requires 30 days notice
MACHINE_LEAD_DAYS   = 180.0  # Tier-2: new machine available after 6 months
ROLLING_WINDOW_DAYS = 30.0   # window for rolling-average utilisation

# ── Workcenter definitions ────────────────────────────────────────────────────
# {wc: (n_machines, dept, hours_per_day, capacity_type)}
WC_CONFIG = {
    "WC_AMILL":    (24, "AUTO",   21, "limited"),
    "WC_AMILL_HS": ( 3, "AUTO",   21, "limited"),
    "WC_MMILL":    (19, "MANUAL", 16, "limited"),
    "WC_LATHE":    ( 7, "MANUAL", 16, "limited"),
    "WC_BW":       ( 4, "MANUAL", 16, "limited"),
    "WC_QC":       ( 9, "MANUAL", 16, "limited"),
    "WC_CMM":      ( 6, "MANUAL", 16, "limited"),
    "WC_3DP":      ( 6, "MANUAL", 16, "limited"),
    "WC_SAW":      ( 3, "MANUAL", 16, "limited"),
    "WC_CONV":     ( 4, "MANUAL", 16, "limited"),
    "WC_REWORK":   ( 1, "MANUAL", 16, "limited"),
    "WC_MISC":     (None, "MANUAL", 16, "unlimited"),
    "WC_OP":       (None, "MANUAL", 16, "unlimited"),
}

# Fixed lead times for unlimited WCs (working days)
FIXED_LT = {"WC_MISC": 2.0, "WC_OP": 5.0}

TRANSPORT_DELAY_WD = 30 / (16 * 60)   # 30 min ≈ 0.031 working days

SIM_ORIGIN   = date.fromisoformat(SIM_START_STR)
SIM_END_DAYS = float((date.fromisoformat(SIM_END_STR) - SIM_ORIGIN).days)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def to_sim(d: date) -> float:
    return float((d - SIM_ORIGIN).days)

def sim_to_date(t: float) -> date:
    return SIM_ORIGIN + timedelta(days=int(t))

def next_workday_sim(t: float) -> float:
    """Advance to next Monday if t falls on a weekend."""
    d = SIM_ORIGIN + timedelta(days=int(t))
    frac = t - int(t)
    while d.weekday() >= 5:
        d += timedelta(days=1)
        frac = 0.0
    return to_sim(d) + frac

def dispatch_key(job: dict) -> float:
    if DISPATCH_RULE == "EDD":  return job["due_sim"]
    if DISPATCH_RULE == "SPT":  return job["proc_days_cur"]
    return job["arrive_wc_time"]   # FIFO

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data…")

df_routing = pd.read_excel(DATA_FILE, sheet_name="Item_Routing", header=0)
df_routing.columns = [
    "item_name", "operation", "setup_days", "run_days",
    "outside_process_fix_lt", "wc", "leadtime", "planned_qtime"
]
df_routing = df_routing.dropna(subset=["item_name"])
df_routing["setup_days"]    = pd.to_numeric(df_routing["setup_days"],    errors="coerce").fillna(0)
df_routing["run_days"]      = pd.to_numeric(df_routing["run_days"],      errors="coerce").fillna(0)
df_routing["planned_qtime"] = pd.to_numeric(df_routing["planned_qtime"], errors="coerce").fillna(0)
df_routing["operation"]     = pd.to_numeric(df_routing["operation"],     errors="coerce")
df_routing = df_routing.dropna(subset=["operation"]).copy()

df_wo = pd.read_excel(DATA_FILE, sheet_name="WOs", header=0)
df_wo.columns = ["wo_num", "item_name", "start_date", "due_date", "ord_qty"]
df_wo = df_wo.dropna(subset=["wo_num"])
df_wo["ord_qty"]    = pd.to_numeric(df_wo["ord_qty"],  errors="coerce").fillna(1).astype(int)
df_wo["start_date"] = pd.to_datetime(df_wo["start_date"])
df_wo["due_date"]   = pd.to_datetime(df_wo["due_date"])

# Build routing dict
routing_dict: dict[str, list[dict]] = {}
for item, grp in df_routing.groupby("item_name"):
    routing_dict[item] = grp.sort_values("operation").to_dict("records")

print(f"Items with routing : {len(routing_dict)}")
print(f"Work orders        : {len(df_wo)}")
items_no_routing = df_wo[~df_wo["item_name"].isin(routing_dict)]["item_name"].unique()
print(f"WOs without routing: {len(items_no_routing)} items → skipped")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  WORKCENTER STATE
# ═══════════════════════════════════════════════════════════════════════════════
class Machine:
    def __init__(self, mid, wc):
        self.id         = mid
        self.wc         = wc
        self.busy       = False
        self.total_busy = 0.0   # accumulated processing time (working days)

class WorkCenter:
    """
    Represents one workcenter.  Tracks machines, queue, utilisation history,
    and capacity-expansion state for GFC decisions.
    """
    def __init__(self, name, n_machines, dept, hours_per_day, cap_type):
        self.name          = name
        self.dept          = dept
        self.cap_type      = cap_type
        self.hours_per_day = hours_per_day    # current value; may increase via Tier-1
        self.base_hours    = hours_per_day    # original value (for reporting)
        self.machines      = [Machine(i, name) for i in range(n_machines)] if n_machines else []
        self.queue: list[dict] = []

        # Statistics
        self.total_ops_done  = 0
        self.queue_time_sum  = 0.0
        self.queue_snaps: list[tuple] = []   # (sim_t, q_len)

        # GFC rolling-utilisation tracking
        # Stores (sim_t, busy_machines_fraction) snapshots for rolling average
        self.util_snaps: list[tuple] = []    # (sim_t, instantaneous busy fraction)

        # GFC capacity-expansion state
        self.alert_active        = False     # currently above UTIL_ALERT
        self.overload_since      = None      # sim_t when rolling util exceeded UTIL_EXPAND
        self.tier1_done          = False     # hours-extension already applied
        self.tier1_effective_at  = None      # sim_t when Tier-1 expansion is effective
        self.tier2_ordered       = False     # new machine already ordered
        self.tier2_effective_at  = None      # sim_t when new machine comes online
        self.tier2_machine_added = False     # True once the machine object is added

    def find_idle(self):
        for m in self.machines:
            if not m.busy:
                return m
        return None

    @property
    def n_machines(self):
        return len(self.machines)

    def instantaneous_util(self) -> float:
        """Fraction of machines currently busy (0–1)."""
        if not self.machines:
            return 0.0
        return sum(1 for m in self.machines if m.busy) / len(self.machines)

    def rolling_util(self, now: float, window: float = ROLLING_WINDOW_DAYS) -> float:
        """
        Rolling average utilisation (0–100%) over the last `window` sim-days,
        computed from stored util_snaps.
        """
        if not self.util_snaps:
            return 0.0
        cutoff = now - window
        recent = [(t, u) for t, u in self.util_snaps if t >= cutoff]
        if not recent:
            recent = self.util_snaps[-1:]
        return 100.0 * sum(u for _, u in recent) / len(recent)


workcenters: dict[str, WorkCenter] = {}
for wc_name, (nm, dept, hpd, cap) in WC_CONFIG.items():
    workcenters[wc_name] = WorkCenter(wc_name, nm if nm else 0, dept, hpd, cap)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  GFC MODULE
# ═══════════════════════════════════════════════════════════════════════════════
expansion_log: list[dict] = []   # all capacity decisions
alert_log:     list[dict] = []   # all utilisation alerts


def gfc_monitor(now: float):
    """
    Periodic GFC check.  Called every MONITOR_INTERVAL sim-days via a scheduled event.

    For each limited workcenter:
      1. Record current rolling utilisation snapshot.
      2. If rolling util > UTIL_ALERT → log alert.
      3. If rolling util > UTIL_EXPAND → start or continue overload timer.
         Once timer exceeds EXPAND_SUSTAIN_DAYS → trigger capacity expansion.
      4. If rolling util drops back below UTIL_EXPAND → reset overload timer.
    """
    for wc_name, wc in workcenters.items():
        if wc.cap_type != "limited":
            continue

        roll_u = wc.rolling_util(now)

        # ── Alert threshold ────────────────────────────────────────────────────
        if roll_u > UTIL_ALERT:
            if not wc.alert_active:
                wc.alert_active = True
                alert_log.append({
                    "sim_t": now,
                    "date":  sim_to_date(now).isoformat(),
                    "wc":    wc_name,
                    "util":  round(roll_u, 1),
                    "msg":   f"ALERT: {wc_name} rolling util {roll_u:.1f}% > {UTIL_ALERT}%"
                })
                print(f"  [GFC {sim_to_date(now)}] ⚠  ALERT  {wc_name}: {roll_u:.1f}%")
        else:
            wc.alert_active = False

        # ── Expansion threshold ────────────────────────────────────────────────
        if roll_u > UTIL_EXPAND:
            if wc.overload_since is None:
                wc.overload_since = now
            elif (now - wc.overload_since) >= EXPAND_SUSTAIN_DAYS:
                _trigger_expansion(wc, now, roll_u)
        else:
            wc.overload_since = None   # reset: not sustained


def _trigger_expansion(wc: WorkCenter, now: float, roll_u: float):
    """
    Decide which tier to apply and schedule the expansion event.
    Tier-1 (hours extension) applies to MANUAL WCs that have headroom.
    Tier-2 (add machine) applies when Tier-1 is exhausted or for AUTO WCs.
    """
    if wc.dept == "MANUAL" and not wc.tier1_done and wc.hours_per_day < 21:
        # ── Tier-1: extend hours ───────────────────────────────────────────────
        old_h       = wc.hours_per_day
        new_h       = 21                              # extend to max (21 h/day)
        extra_h     = new_h - old_h
        n_mach      = wc.n_machines
        monthly_cost = extra_h * n_mach * 1_500      # €1,500 / machine / extra-hr / month
        effective_t  = now + EXPAND_NOTICE_DAYS

        wc.tier1_done         = True
        wc.tier1_effective_at = effective_t
        # hours_per_day will be updated when the event fires
        push(effective_t, EV_EXPAND_HOURS, {
            "wc": wc.name, "new_hours": new_h, "decision_t": now
        })

        msg = (f"TIER-1 decision: {wc.name} hours {old_h}→{new_h}h/day "
               f"({n_mach} machines × +{extra_h}h × €1,500/mo = €{monthly_cost:,}/mo). "
               f"Effective {sim_to_date(effective_t)} (30-day notice).")
        print(f"  [GFC {sim_to_date(now)}] 🔧 {msg}")
        expansion_log.append({
            "sim_t": now, "date": sim_to_date(now).isoformat(),
            "wc": wc.name, "tier": 1,
            "old_hours": old_h, "new_hours": new_h,
            "effective_date": sim_to_date(effective_t).isoformat(),
            "monthly_cost_eur": monthly_cost,
            "capex_eur": 0, "annual_opex_eur": 0,
            "trigger_util": round(roll_u, 1),
        })

    elif not wc.tier2_ordered:
        # ── Tier-2: add one machine ────────────────────────────────────────────
        effective_t = now + MACHINE_LEAD_DAYS
        capex       = 500_000
        annual_opex = 100_000

        wc.tier2_ordered      = True
        wc.tier2_effective_at = effective_t
        push(effective_t, EV_EXPAND_MACHINE, {
            "wc": wc.name, "decision_t": now
        })

        msg = (f"TIER-2 decision: {wc.name} +1 machine. "
               f"€{capex:,} CAPEX + €{annual_opex:,}/yr OPEX. "
               f"Online {sim_to_date(effective_t)} (6-month lead time).")
        print(f"  [GFC {sim_to_date(now)}] 🏭 {msg}")
        expansion_log.append({
            "sim_t": now, "date": sim_to_date(now).isoformat(),
            "wc": wc.name, "tier": 2,
            "old_hours": wc.hours_per_day, "new_hours": wc.hours_per_day,
            "effective_date": sim_to_date(effective_t).isoformat(),
            "monthly_cost_eur": round(annual_opex / 12),
            "capex_eur": capex, "annual_opex_eur": annual_opex,
            "trigger_util": round(roll_u, 1),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EVENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
EV_RELEASE      = 0   # WO released to shopfloor
EV_OP_START     = 1   # (unused — start is implicit in arrive_at_wc)
EV_OP_DONE      = 2   # Operation finished
EV_FIXED_DONE   = 3   # Fixed-LT step completed
EV_GFC_CHECK    = 4   # Periodic GFC monitoring event
EV_EXPAND_HOURS = 5   # Tier-1 expansion becomes effective (hours update)
EV_EXPAND_MACHINE = 6 # Tier-2 expansion becomes effective (machine added)

events: list = []
_ctr = 0

def push(t: float, etype: int, payload: dict):
    global _ctr
    heapq.heappush(events, (t, _ctr, etype, payload))
    _ctr += 1


def start_op_on_machine(wc: WorkCenter, m: Machine, job: dict, now: float):
    m.busy   = True
    finish   = now + job["proc_days_cur"]
    push(finish, EV_OP_DONE, {
        "wc": wc.name, "machine_id": m.id,
        "job": job, "start_time": now, "finish_time": finish
    })


def arrive_at_wc(wc: WorkCenter, job: dict, now: float):
    job["arrive_wc_time"] = now
    if wc.cap_type == "unlimited":
        delay  = FIXED_LT.get(wc.name, 1.0)
        finish = now + delay
        push(finish, EV_FIXED_DONE, {
            "wc": wc.name, "job": job,
            "start_time": now, "finish_time": finish
        })
    else:
        idle = wc.find_idle()
        if idle:
            job["queue_time_cur"] = 0.0
            start_op_on_machine(wc, idle, job, now)
        else:
            wc.queue.append(job)


# Per-operation and per-WO result accumulators
wo_completed: list[dict] = []
op_records:   list[dict] = []


def advance_job(job: dict, now: float, op_rec: dict | None = None):
    """Move job to next operation or mark it complete."""
    if op_rec:
        op_records.append(op_rec)

    ops    = job["ops"]
    op_idx = job["op_idx"] + 1
    job["op_idx"] = op_idx

    if op_idx >= len(ops):
        wo_completed.append({
            "wo_num":      job["wo_num"],
            "item_name":   job["item_name"],
            "ord_qty":     job["ord_qty"],
            "release_sim": job["release_sim"],
            "due_sim":     job["due_sim"],
            "start_sim":   job["release_sim"],
            "finish_sim":  now,
            "lead_time":   now - job["release_sim"],
            "tardy":       now > job["due_sim"],
            "tardiness":   max(0.0, now - job["due_sim"]),
        })
        return

    op      = ops[op_idx]
    wc_name = op["wc"]
    proc    = op["setup_days"] + op["run_days"] * job["ord_qty"]
    job["proc_days_cur"] = proc
    job["planned_q_cur"] = op["planned_qtime"]
    job["cur_wc"]        = wc_name

    t_arrive = next_workday_sim(now + TRANSPORT_DELAY_WD)
    wc = workcenters.get(wc_name)
    if wc is None:
        push(t_arrive + 1.0, EV_FIXED_DONE, {
            "wc": wc_name, "job": job,
            "start_time": t_arrive, "finish_time": t_arrive + 1.0
        })
    else:
        arrive_at_wc(wc, job, t_arrive)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SCHEDULE INITIAL EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Work order releases
released = 0
for _, wo in df_wo.iterrows():
    if wo["item_name"] not in routing_dict:
        continue
    ops     = routing_dict[wo["item_name"]]
    rel_t   = next_workday_sim(to_sim(wo["start_date"].date()))
    job = {
        "wo_num":      str(wo["wo_num"]),
        "item_name":   str(wo["item_name"]),
        "ord_qty":     int(wo["ord_qty"]),
        "due_sim":     to_sim(wo["due_date"].date()),
        "release_sim": rel_t,
        "ops":         ops,
        "op_idx":      -1,
    }
    push(rel_t, EV_RELEASE, {"job": job})
    released += 1

# GFC periodic check events
t_gfc = MONITOR_INTERVAL
while t_gfc <= SIM_END_DAYS:
    push(t_gfc, EV_GFC_CHECK, {"t": t_gfc})
    t_gfc += MONITOR_INTERVAL

print(f"Work orders scheduled : {released}")
print(f"GFC check events      : {int(SIM_END_DAYS / MONITOR_INTERVAL)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
SNAP_INTERVAL = 1.0
print("\nRunning simulation…")
processed_events = 0

while events:
    now, _, etype, payload = heapq.heappop(events)
    if now > SIM_END_DAYS:
        break
    processed_events += 1

    # Record instantaneous utilisation snapshots for rolling average
    for wc in workcenters.values():
        if wc.cap_type == "limited":
            if not wc.util_snaps or (now - wc.util_snaps[-1][0]) >= SNAP_INTERVAL:
                wc.util_snaps.append((now, wc.instantaneous_util()))
            if not wc.queue_snaps or (now - wc.queue_snaps[-1][0]) >= SNAP_INTERVAL:
                wc.queue_snaps.append((now, len(wc.queue)))

    # ── WO RELEASE ──────────────────────────────────────────────────────────
    if etype == EV_RELEASE:
        advance_job(payload["job"], now)

    # ── OPERATION DONE ───────────────────────────────────────────────────────
    elif etype == EV_OP_DONE:
        wc_name = payload["wc"]
        wc      = workcenters[wc_name]
        m       = wc.machines[payload["machine_id"]]
        job     = payload["job"]
        t0      = payload["start_time"]

        m.busy        = False
        m.total_busy += job["proc_days_cur"]
        wc.total_ops_done += 1
        q_time = t0 - job["arrive_wc_time"]
        wc.queue_time_sum += q_time

        rec = {
            "wo_num":      job["wo_num"],
            "item_name":   job["item_name"],
            "wc":          wc_name,
            "operation":   job["ops"][job["op_idx"]]["operation"],
            "ord_qty":     job["ord_qty"],
            "proc_days":   job["proc_days_cur"],
            "planned_q":   job["planned_q_cur"],
            "arrive_wc":   job["arrive_wc_time"],
            "start_time":  t0,
            "finish_time": now,
            "queue_time":  q_time,
            "due_sim":     job["due_sim"],
        }

        # Serve next queued job (dispatch rule)
        if wc.queue:
            wc.queue.sort(key=dispatch_key)
            nxt = wc.queue.pop(0)
            nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
            start_op_on_machine(wc, m, nxt, now)

        advance_job(job, now, rec)

    # ── FIXED-LT DONE ────────────────────────────────────────────────────────
    elif etype == EV_FIXED_DONE:
        job = payload["job"]
        rec = {
            "wo_num":      job["wo_num"],
            "item_name":   job["item_name"],
            "wc":          payload["wc"],
            "operation":   job["ops"][job["op_idx"]]["operation"] if job["op_idx"] < len(job["ops"]) else -1,
            "ord_qty":     job["ord_qty"],
            "proc_days":   payload["finish_time"] - payload["start_time"],
            "planned_q":   0.0,
            "arrive_wc":   payload["start_time"],
            "start_time":  payload["start_time"],
            "finish_time": payload["finish_time"],
            "queue_time":  0.0,
            "due_sim":     job["due_sim"],
        }
        advance_job(job, payload["finish_time"], rec)

    # ── GFC PERIODIC CHECK ───────────────────────────────────────────────────
    elif etype == EV_GFC_CHECK:
        gfc_monitor(now)

    # ── TIER-1: HOURS EXTENSION EFFECTIVE ────────────────────────────────────
    elif etype == EV_EXPAND_HOURS:
        wc_name  = payload["wc"]
        new_h    = payload["new_hours"]
        wc       = workcenters[wc_name]
        old_h    = wc.hours_per_day
        wc.hours_per_day = new_h
        print(f"  [SIM  {sim_to_date(now)}] ✅ Tier-1 effective: "
              f"{wc_name} now running {new_h}h/day (was {old_h}h)")
        # Note: hours_per_day affects available-capacity calculations in reporting;
        # it does not change machine free_at (jobs in progress are unaffected).
        # New jobs arriving from this point benefit from the extended day, reflected
        # implicitly since processing times are expressed in working days per the
        # assignment spec (proc time already independent of hours/day).
        # The impact is captured in utilisation: available capacity denominator increases.

    # ── TIER-2: NEW MACHINE ONLINE ────────────────────────────────────────────
    elif etype == EV_EXPAND_MACHINE:
        wc_name = payload["wc"]
        wc      = workcenters[wc_name]
        if not wc.tier2_machine_added:
            new_mid = wc.n_machines          # next sequential id
            new_m   = Machine(new_mid, wc_name)
            wc.machines.append(new_m)
            wc.tier2_machine_added = True
            print(f"  [SIM  {sim_to_date(now)}] ✅ Tier-2 effective: "
                  f"{wc_name} +1 machine (now {wc.n_machines} total)")
            # Immediately try to dispatch a queued job to the new machine
            if wc.queue:
                wc.queue.sort(key=dispatch_key)
                nxt = wc.queue.pop(0)
                nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                start_op_on_machine(wc, new_m, nxt, now)

print(f"Simulation done. Events processed: {processed_events}")
print(f"WOs completed: {len(wo_completed)} / {released}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  RESULTS DATAFRAMES
# ═══════════════════════════════════════════════════════════════════════════════
df_wo_done = pd.DataFrame(wo_completed)
df_ops     = pd.DataFrame(op_records)

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  STEP 3 RESULTS  |  dispatch: {DISPATCH_RULE}  |  GFC active")
print("="*70)

if len(df_wo_done):
    n_tardy = int(df_wo_done["tardy"].sum())
    pct_ota = 100 * (1 - n_tardy / len(df_wo_done))
    avg_lt  = df_wo_done["lead_time"].mean()
    p95_lt  = df_wo_done["lead_time"].quantile(0.95)
    print(f"\n  WOs completed     : {len(df_wo_done)} / {released}")
    print(f"  Mean lead time    : {avg_lt:.2f} days")
    print(f"  P95  lead time    : {p95_lt:.2f} days")
    print(f"  On-time delivery  : {pct_ota:.1f}%  ({len(df_wo_done)-n_tardy}/{len(df_wo_done)})")
    print(f"  Tardy WOs         : {n_tardy}")
    if n_tardy:
        print(f"  Avg tardiness     : {df_wo_done.loc[df_wo_done['tardy'],'tardiness'].mean():.2f} days")
else:
    pct_ota = 0.0; avg_lt = 0.0; p95_lt = 0.0

# ── Per-workcenter table ───────────────────────────────────────────────────────
print(f"\n  {'WC':<14} {'Mach':>5} {'Hrs':>4} {'Avail(d)':>10} {'Busy(d)':>10} "
      f"{'Util%':>7} {'Ops':>6} {'MeanQ(d)':>9} {'MaxQ(d)':>8}")
print("  " + "-"*72)

util_data = {}
for wc_name, wc in sorted(workcenters.items()):
    if wc.cap_type == "unlimited" or wc.n_machines == 0:
        continue
    avail = wc.n_machines * wc.hours_per_day * SIM_END_DAYS / 24.0
    busy  = sum(m.total_busy for m in wc.machines)
    util  = 100.0 * busy / avail if avail > 0 else 0.0

    wc_ops = df_ops[df_ops["wc"] == wc_name] if len(df_ops) else pd.DataFrame()
    mean_q = wc_ops["queue_time"].mean() if len(wc_ops) else 0.0
    max_q  = wc_ops["queue_time"].max()  if len(wc_ops) else 0.0

    util_data[wc_name] = {
        "util": util, "busy": busy, "avail": avail,
        "ops":  wc.total_ops_done, "mean_q": mean_q, "max_q": max_q,
        "hours": wc.hours_per_day,
    }
    flag = " ◄ BOTTLENECK" if util > 85 else (" ◄ WARNING" if util > 70 else "")
    print(f"  {wc_name:<14} {wc.n_machines:>5} {wc.hours_per_day:>4} {avail:>10.1f} "
          f"{busy:>10.2f} {util:>6.1f}% {wc.total_ops_done:>6} "
          f"{mean_q:>9.2f} {max_q:>8.2f}{flag}")

# ── GFC expansion summary ─────────────────────────────────────────────────────
print(f"\n  ── GFC Capacity Expansion Decisions ({'none' if not expansion_log else len(expansion_log)}) ──")
if expansion_log:
    total_monthly = 0; total_capex = 0
    for e in expansion_log:
        print(f"  {e['date']}  Tier-{e['tier']}  {e['wc']:<14}  "
              f"effective {e['effective_date']}  "
              f"trigger util {e['trigger_util']}%  "
              f"€{e['monthly_cost_eur']:,}/mo  CAPEX €{e['capex_eur']:,}")
        total_monthly += e["monthly_cost_eur"]
        total_capex   += e["capex_eur"]
    print(f"\n  Total extra monthly cost : €{total_monthly:,}/mo")
    print(f"  Total CAPEX              : €{total_capex:,}")
else:
    print("  No capacity expansions triggered during simulation horizon.")

print(f"\n  ── GFC Alerts fired: {len(alert_log)} ──")
for a in alert_log:
    print(f"  {a['date']}  {a['wc']:<14}  rolling util = {a['util']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def sd(t: float):
    """sim-days → datetime for matplotlib."""
    return datetime.combine(sim_to_date(t), datetime.min.time())


fig, axes = plt.subplots(3, 2, figsize=(16, 14))
title_str = (
    f"Step 3 — GFC Integrated  |  Dispatch: {DISPATCH_RULE}\n"
    f"WOs completed: {len(df_wo_done)}/{released}   "
    f"On-time: {pct_ota:.1f}%   "
    f"Mean LT: {avg_lt:.1f} d   "
    f"Expansions: {len(expansion_log)}"
)
fig.suptitle(title_str, fontsize=11, fontweight="bold")

# ── 1. Per-WC utilisation bar chart with expansion markers ───────────────────
ax = axes[0, 0]
wc_sorted = sorted(util_data.keys())
utils     = [util_data[w]["util"] for w in wc_sorted]
colors    = ["red" if u > 85 else ("orange" if u > 70 else "steelblue") for u in utils]
bars = ax.barh(wc_sorted, utils, color=colors, edgecolor="white", height=0.6)
ax.axvline(85, color="red",    linestyle="--", linewidth=1, label="85% alert")
ax.axvline(70, color="orange", linestyle="--", linewidth=0.8, label="70% warning")
ax.set_title("Workcenter Utilisation (%) — full horizon")
ax.set_xlabel("Utilisation %")
ax.set_xlim(0, 110)
ax.legend(fontsize=8)
for bar, u in zip(bars, utils):
    ax.text(u + 0.5, bar.get_y() + bar.get_height()/2,
            f"{u:.1f}%", va="center", fontsize=7)
# Mark WCs that received expansion
expanded_wcs = {e["wc"] for e in expansion_log}
for i, wc_name in enumerate(wc_sorted):
    if wc_name in expanded_wcs:
        ax.text(3, i, "⬆ expanded", va="center", fontsize=7, color="darkgreen")
ax.grid(True, alpha=0.3, axis="x")

# ── 2. Rolling utilisation over time — top-3 most loaded WCs ─────────────────
ax = axes[0, 1]
top3 = sorted(util_data, key=lambda w: util_data[w]["util"], reverse=True)[:3]
pal  = ["red", "darkorange", "steelblue"]
for wc_name, col in zip(top3, pal):
    snaps = workcenters[wc_name].util_snaps
    if not snaps:
        continue
    ts, us = zip(*snaps)
    dts = [sd(t) for t in ts]
    pct = [100 * u for u in us]
    ax.plot(dts, pct, color=col, linewidth=0.8, alpha=0.8, label=wc_name)

ax.axhline(UTIL_ALERT,  color="red",    linestyle="--", linewidth=1,
           label=f"{UTIL_ALERT}% alert")
ax.axhline(UTIL_EXPAND, color="darkred", linestyle=":",  linewidth=1,
           label=f"{UTIL_EXPAND}% expand")

# Overlay expansion event vertical lines
for e in expansion_log:
    ax.axvline(datetime.combine(date.fromisoformat(e["effective_date"]),
                                datetime.min.time()),
               color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7)

ax.set_title("Instantaneous Utilisation — Top-3 WCs")
ax.set_ylabel("% machines busy")
ax.set_ylim(0, 110)
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
ax.grid(True, alpha=0.3)

# ── 3. WO lead time distribution ─────────────────────────────────────────────
ax = axes[1, 0]
if len(df_wo_done):
    ax.hist(df_wo_done["lead_time"], bins=50, color="slateblue",
            edgecolor="white", linewidth=0.3)
    ax.axvline(avg_lt, color="red",        linestyle="--",
               label=f"Mean={avg_lt:.1f}d")
    ax.axvline(p95_lt, color="darkorange", linestyle="--",
               label=f"P95={p95_lt:.1f}d")
    ax.legend(fontsize=8)
ax.set_title("WO End-to-End Lead Time Distribution")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.3)

# ── 4. Queue length over time — top-3 WCs ────────────────────────────────────
ax = axes[1, 1]
for wc_name, col in zip(top3, pal):
    snaps = workcenters[wc_name].queue_snaps
    if not snaps:
        continue
    ts, qs = zip(*snaps)
    dts    = [sd(t) for t in ts]
    ax.plot(dts, qs, color=col, linewidth=0.8, label=wc_name)
for e in expansion_log:
    ax.axvline(datetime.combine(date.fromisoformat(e["effective_date"]),
                                datetime.min.time()),
               color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7)
ax.set_title("Queue Length Over Time — Top-3 WCs")
ax.set_ylabel("Jobs in queue")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
ax.grid(True, alpha=0.3)

# ── 5. Mean queue time: simulated vs MRP planned ─────────────────────────────
ax = axes[2, 0]
mean_qs   = [util_data[w]["mean_q"] for w in wc_sorted]
planned_q = []
for w in wc_sorted:
    wc_ops = df_ops[df_ops["wc"] == w] if len(df_ops) else pd.DataFrame()
    planned_q.append(wc_ops["planned_q"].mean() if len(wc_ops) else 0.0)
x = range(len(wc_sorted))
ax.bar(x,               mean_qs,   width=0.4, label="Simulated",    color="steelblue", alpha=0.8)
ax.bar([i+0.4 for i in x], planned_q, width=0.4, label="MRP planned", color="seagreen",  alpha=0.8)
ax.set_xticks([i+0.2 for i in x])
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
    ax.axhline(95,      color="red",  linestyle="--", linewidth=1,   label="95% target")
    ax.axhline(pct_ota, color="navy", linestyle=":",  linewidth=1,
               label=f"Overall {pct_ota:.1f}%")
    for e in expansion_log:
        ax.axvline(datetime.combine(date.fromisoformat(e["effective_date"]),
                                    datetime.min.time()),
                   color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7)
ax.set_ylim(0, 105)
ax.set_title("Rolling On-Time Delivery Rate (50-WO window)")
ax.set_ylabel("On-time %")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT / "step3_simulation.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════════════
if len(df_wo_done):
    p = OUT / "step3_wo_results.csv"
    df_wo_done.to_csv(p, index=False)
    print(f"WO results    → {p}")

if len(df_ops):
    p = OUT / "step3_op_results.csv"
    df_ops.to_csv(p, index=False)
    print(f"Op results    → {p}")

if expansion_log:
    df_exp = pd.DataFrame(expansion_log)
    p = OUT / "step3_expansion_log.csv"
    df_exp.to_csv(p, index=False)
    print(f"Expansion log → {p}")

if alert_log:
    df_alerts = pd.DataFrame(alert_log)
    p = OUT / "step3_alert_log.csv"
    df_alerts.to_csv(p, index=False)
    print(f"Alert log     → {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  CONCEPTUAL DESIGN VALIDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  CONCEPTUAL DESIGN VALIDATION — STEP 3")
print("="*70)

bottlenecks = [w for w, d in util_data.items() if d["util"] > 85]
warnings    = [w for w, d in util_data.items() if 70 < d["util"] <= 85]
top_wc      = max(util_data, key=lambda w: util_data[w]["util"]) if util_data else "N/A"

tier1_decisions = [e for e in expansion_log if e["tier"] == 1]
tier2_decisions = [e for e in expansion_log if e["tier"] == 2]
total_capex     = sum(e["capex_eur"] for e in expansion_log)
total_monthly   = sum(e["monthly_cost_eur"] for e in expansion_log)

amill_hs_u = util_data.get("WC_AMILL_HS", {}).get("util", 0.0)
mmill_u    = util_data.get("WC_MMILL",    {}).get("util", 0.0)

print(f"""
  [B1] WC_AMILL_HS (Auriga) utilisation = {amill_hs_u:.1f}%
       {"→ CONFIRMED bottleneck candidate: alert expected" if amill_hs_u > UTIL_ALERT
        else "→ NOT a bottleneck at this demand level"}

  [B2] On-time delivery with EDD = {pct_ota:.1f}%
       (Re-run with DISPATCH_RULE='FIFO' or 'SPT' for comparison)

  [B3] Tier-1 expansions (hours): {len(tier1_decisions)} — {[e['wc'] for e in tier1_decisions]}
       Tier-2 expansions (machine): {len(tier2_decisions)} — {[e['wc'] for e in tier2_decisions]}
       Total CAPEX: €{total_capex:,}   Extra opex: €{total_monthly:,}/mo
       {"→ Tier-1 sufficient: no Tier-2 machines added" if not tier2_decisions
        else "→ Tier-2 triggered: hours-extension alone insufficient for sustained load"}

  [B4] GFC alert sensitivity:
       Alerts fired: {len(alert_log)} (threshold {UTIL_ALERT}%)
       Earliest alert: {alert_log[0]['date'] if alert_log else 'N/A'} on {alert_log[0]['wc'] if alert_log else 'N/A'}
       {"→ GFC lead time sufficient for Tier-1 notice window" if len(alert_log) > 0
        else "→ No alerts; all WCs within guideline"}

  Most loaded WC: {top_wc}  ({util_data.get(top_wc, {}).get('util', 0):.1f}%)
  Bottlenecks (>85%): {bottlenecks if bottlenecks else 'None'}
  Warnings   (>70%): {warnings  if warnings  else 'None'}
  WC_MMILL utilisation: {mmill_u:.1f}%  {"→ secondary bottleneck" if mmill_u > 70 else ""}

  Next step (Step 4):
      Add scenario analysis (demand growth), sensitivity (90% uptime),
      and comparative dispatching-rule experiments.
""")
