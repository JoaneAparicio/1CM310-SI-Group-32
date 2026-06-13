"""
gfc.py — SDG Shopfloor Simulation
===================================
Goods Flow Control (GFC) module.

Expansion rules (MANUAL workcenters):
  +---------------------------------------------------------------------+
  | Zone          | Threshold  | Sustained      | Action                |
  +---------------------------------------------------------------------+
  | Warning zone  | > 70%      | WARNING_SUSTAIN | +2 h/day (cap 21)   |
  | Alert zone    | > 85%      | ALERT_SUSTAIN   | Jump to 21h          |
  | Exhausted     | > 85%      | ALERT_SUSTAIN   | +1 machine (Tier-2)  |
  |   (already 21h)            |                 |                      |
  +---------------------------------------------------------------------+

WC_BW exception: if already at 21h and first Tier-2 machine is online,
a second machine may be added when util is still > UTIL_ALERT.

AUTO workcenters cannot extend hours -> alert zone goes straight to Tier-2.

Costs:
  Tier-1: 1500 EUR / machine / extra-hour / month (cumulative vs base 16 h)
  Tier-2: 500,000 EUR CAPEX + 100,000 EUR/yr OPEX, 6-month lead time.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Callable

import config
from workcenter import WorkCenter

# Shared logs
expansion_log: list[dict] = []
alert_log:     list[dict] = []


def _sim_to_date(t: float) -> date:
    return config.SIM_ORIGIN + timedelta(days=int(t))


def gfc_monitor(
    now: float,
    workcenters: dict[str, WorkCenter],
    push: Callable,
    ev_expand_hours: int,
    ev_expand_machine: int,
) -> None:
    """
    Periodic GFC check. Called every MONITOR_INTERVAL sim-days.

    Two independent sustained-overload timers per WC:
      warning_since : tracks time continuously above UTIL_WARNING
      alert_since   : tracks time continuously above UTIL_ALERT

    Alert zone always takes priority over warning zone.
    Timers reset when utilisation drops back below the respective threshold,
    or immediately after a capacity decision is taken.
    """
    if now < config.WARMUP_DAYS:
        return   # no decisions during warm-up

    for wc_name, wc in workcenters.items():
        if wc.cap_type != "limited":
            continue

        roll_u = wc.rolling_util(now)

        # Log alert (once per overload episode)
        if roll_u > config.UTIL_ALERT:
            if not wc.alert_active:
                wc.alert_active = True
                entry = {
                    "sim_t": now,
                    "date":  _sim_to_date(now).isoformat(),
                    "wc":    wc_name,
                    "util":  round(roll_u, 1),
                    "msg":   f"ALERT: {wc_name} {roll_u:.1f}% > {config.UTIL_ALERT}%",
                }
                alert_log.append(entry)
                print(f"  [GFC {_sim_to_date(now)}] ALERT   {wc_name}: {roll_u:.1f}%")
        else:
            wc.alert_active = False

        if roll_u > config.UTIL_WARNING and roll_u <= config.UTIL_ALERT:
            if not wc.warning_active:
                wc.warning_active = True
                print(f"  [GFC {_sim_to_date(now)}] WARNING {wc_name}: {roll_u:.1f}%")
        else:
            wc.warning_active = False

        # Alert timer (> UTIL_ALERT) - highest priority
        if roll_u > config.UTIL_ALERT:
            if wc.alert_since is None:
                wc.alert_since = now
            elif (now - wc.alert_since) >= config.ALERT_SUSTAIN_DAYS:
                if config.GFC_EXPANSIONS_ENABLED:
                    _trigger_alert_expansion(wc, now, roll_u, push,
                                             ev_expand_hours, ev_expand_machine)
                wc.alert_since   = None
                wc.warning_since = None
        else:
            wc.alert_since = None

        # Warning timer (> UTIL_WARNING but <= UTIL_ALERT)
        if config.UTIL_WARNING < roll_u <= config.UTIL_ALERT:
            if wc.warning_since is None:
                wc.warning_since = now
            elif (now - wc.warning_since) >= config.WARNING_SUSTAIN_DAYS:
                if config.GFC_EXPANSIONS_ENABLED:
                    _trigger_warning_expansion(wc, now, roll_u, push,
                                               ev_expand_hours)
                wc.warning_since = None
        else:
            if roll_u <= config.UTIL_WARNING:
                wc.warning_since = None


# ---------------------------------------------------------------------------
# Expansion helpers
# ---------------------------------------------------------------------------

def _pending_tier1(wc: WorkCenter, now: float) -> bool:
    """True if a Tier-1 hours-extension event is already scheduled."""
    return wc.tier1_effective_at is not None and wc.tier1_effective_at > now


def _apply_hours_extension(
    wc: WorkCenter,
    now: float,
    new_h: int,
    roll_u: float,
    push: Callable,
    ev_expand_hours: int,
    label: str,
) -> None:
    """Schedule a Tier-1 hours-extension event and log it."""
    old_h         = wc.hours_per_day
    n_mach        = wc.n_machines
    extra_h_total = new_h - wc.base_hours
    monthly_cost  = extra_h_total * n_mach * 1_500
    effective_t   = now + config.EXPAND_NOTICE_DAYS

    wc.tier1_effective_at = effective_t

    push(effective_t, ev_expand_hours, {
        "wc": wc.name, "new_hours": new_h, "decision_t": now
    })

    msg = (
        f"TIER-1 {label}: {wc.name}  {old_h} -> {new_h} h/day  "
        f"[{n_mach} machines, cumul. +{extra_h_total}h -> {monthly_cost:,} EUR/mo].  "
        f"Effective {_sim_to_date(effective_t)}."
    )
    icon = "T1" if new_h < 21 else "T1-MAX"
    print(f"  [GFC {_sim_to_date(now)}] {icon} {msg}")

    expansion_log.append({
        "sim_t":             now,
        "date":              _sim_to_date(now).isoformat(),
        "wc":                wc.name,
        "tier":              1,
        "label":             label,
        "old_hours":         old_h,
        "new_hours":         new_h,
        "extra_hours_total": extra_h_total,
        "effective_date":    _sim_to_date(effective_t).isoformat(),
        "monthly_cost_eur":  monthly_cost,
        "capex_eur":         0,
        "annual_opex_eur":   0,
        "trigger_util":      round(roll_u, 1),
    })


def _apply_machine(
    wc: WorkCenter,
    now: float,
    roll_u: float,
    push: Callable,
    ev_expand_machine: int,
) -> None:
    """Schedule the first Tier-2 machine addition and log it."""
    effective_t = now + config.MACHINE_LEAD_DAYS
    capex       = 500_000
    annual_opex = 100_000

    wc.tier2_ordered      = True
    wc.tier2_effective_at = effective_t

    push(effective_t, ev_expand_machine, {
        "wc": wc.name, "decision_t": now, "second": False,
    })

    msg = (
        f"TIER-2 (+1 machine): {wc.name}.  "
        f"{capex:,} EUR CAPEX + {annual_opex:,} EUR/yr OPEX.  "
        f"Online {_sim_to_date(effective_t)} (6-month lead time)."
    )
    print(f"  [GFC {_sim_to_date(now)}] T2 {msg}")

    expansion_log.append({
        "sim_t":             now,
        "date":              _sim_to_date(now).isoformat(),
        "wc":                wc.name,
        "tier":              2,
        "label":             "machine",
        "old_hours":         wc.hours_per_day,
        "new_hours":         wc.hours_per_day,
        "extra_hours_total": wc.hours_per_day - wc.base_hours,
        "effective_date":    _sim_to_date(effective_t).isoformat(),
        "monthly_cost_eur":  round(annual_opex / 12),
        "capex_eur":         capex,
        "annual_opex_eur":   annual_opex,
        "trigger_util":      round(roll_u, 1),
    })


def _apply_machine_second(
    wc: WorkCenter,
    now: float,
    roll_u: float,
    push: Callable,
    ev_expand_machine: int,
) -> None:
    """
    Schedule a second Tier-2 machine for WC_BW (exception to the 1-machine-max rule).
    Prerequisites: WC_BW at 21 h/day, first extra machine online, util still > UTIL_ALERT.
    """
    effective_t = now + config.MACHINE_LEAD_DAYS
    capex       = 500_000
    annual_opex = 100_000

    wc.tier2_second_ordered = True

    push(effective_t, ev_expand_machine, {
        "wc": wc.name, "decision_t": now, "second": True,
    })

    msg = (
        f"TIER-2 (2nd machine, WC_BW exception): {wc.name}.  "
        f"{capex:,} EUR CAPEX + {annual_opex:,} EUR/yr OPEX.  "
        f"Online {_sim_to_date(effective_t)} (6-month lead time)."
    )
    print(f"  [GFC {_sim_to_date(now)}] T2-BW2 {msg}")

    expansion_log.append({
        "sim_t":             now,
        "date":              _sim_to_date(now).isoformat(),
        "wc":                wc.name,
        "tier":              2,
        "label":             "2nd machine (BW exception)",
        "old_hours":         wc.hours_per_day,
        "new_hours":         wc.hours_per_day,
        "extra_hours_total": wc.hours_per_day - wc.base_hours,
        "effective_date":    _sim_to_date(effective_t).isoformat(),
        "monthly_cost_eur":  round(annual_opex / 12),
        "capex_eur":         capex,
        "annual_opex_eur":   annual_opex,
        "trigger_util":      round(roll_u, 1),
    })


def _trigger_warning_expansion(
    wc: WorkCenter,
    now: float,
    roll_u: float,
    push: Callable,
    ev_expand_hours: int,
) -> None:
    """
    Warning-zone action: +2 h/day (capped at 21).
    Only for MANUAL WCs with hours headroom.
    AUTO WCs: no action from warning zone (cannot extend hours).
    """
    if wc.dept != "MANUAL" or wc.hours_per_day >= 21:
        return
    if _pending_tier1(wc, now):
        return

    step  = 2
    new_h = min(wc.hours_per_day + step, 21)
    _apply_hours_extension(wc, now, new_h, roll_u, push, ev_expand_hours,
                           label=f"+{new_h - wc.hours_per_day}h (warning)")


def _trigger_alert_expansion(
    wc: WorkCenter,
    now: float,
    roll_u: float,
    push: Callable,
    ev_expand_hours: int,
    ev_expand_machine: int,
) -> None:
    """
    Alert-zone action:
      MANUAL + hours < 21  -> jump straight to 21 h/day (Tier-1).
      MANUAL at 21h / AUTO -> +1 machine (Tier-2, once per WC).
      WC_BW exception      -> 2nd machine if 21h + first machine online + still >85%.
    """
    if wc.dept == "MANUAL" and wc.hours_per_day < 21:
        if _pending_tier1(wc, now):
            return
        _apply_hours_extension(wc, now, 21, roll_u, push, ev_expand_hours,
                               label="->21h (alert)")

    elif not wc.tier2_ordered:
        # First extra machine for any WC
        _apply_machine(wc, now, roll_u, push, ev_expand_machine)

    elif (
        wc.name == "WC_BW"
        and wc.hours_per_day >= 21
        and wc.tier2_machine_added          # first machine already online
        and not wc.tier2_second_ordered     # second machine not yet ordered
    ):
        # WC_BW exception: second extra machine
        _apply_machine_second(wc, now, roll_u, push, ev_expand_machine)
