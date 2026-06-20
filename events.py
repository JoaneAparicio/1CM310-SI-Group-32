"""
events.py -- SDG Shopfloor Simulation
======================================
Discrete-event engine.

Optimisations vs original:
  - wc.queue is now a heapq: dispatch is O(log n) instead of O(n log n) sort + O(n) pop(0).
  - Snapshot loop only runs when sim time crosses the next SNAP_INTERVAL boundary,
    not on every event (removes O(n_wcs) overhead from the hot path).
  - schedule_releases uses itertuples instead of iterrows (~5x faster row access).
  - start_op_on_machine / OP_DONE use wc.mark_busy/mark_idle to maintain the
    WorkCenter._idle deque and _n_busy counter correctly.
"""

from __future__ import annotations
import heapq
from datetime import date, timedelta

import config
from workcenter import WorkCenter, Machine
import gfc as GFC


# Event-type constants
EV_RELEASE        = 0
EV_OP_DONE        = 2
EV_FIXED_DONE     = 3
EV_GFC_CHECK      = 4
EV_EXPAND_HOURS   = 5
EV_EXPAND_MACHINE = 6

# Main event heap
_events = []
_ctr    = 0   # tie-breaker for event heap

# Per-WC dispatch queue tie-breaker (separate counter)
_q_ctr  = 0


def push(t, etype, payload):
    """Insert a new event into the priority queue."""
    global _ctr
    heapq.heappush(_events, (t, _ctr, etype, payload))
    _ctr += 1


def to_sim(d):
    return float((d - config.SIM_ORIGIN).days)


def sim_to_date(t):
    return config.SIM_ORIGIN + timedelta(days=int(t))


def next_workday_sim(t):
    """Advance to the next Monday if t falls on a weekend."""
    d    = config.SIM_ORIGIN + timedelta(days=int(t))
    frac = t - int(t)
    while d.weekday() >= 5:
        d    += timedelta(days=1)
        frac  = 0.0
    return to_sim(d) + frac


def dispatch_key(job):
    if config.DISPATCH_RULE == "EDD":
        return job["due_sim"]
    if config.DISPATCH_RULE == "SPT":
        return job["proc_days_cur"]
    return job["arrive_wc_time"]   # FIFO


def _q_push(wc, job):
    """Push a job onto the WC heap queue. O(log n)."""
    global _q_ctr
    heapq.heappush(wc.queue, (dispatch_key(job), _q_ctr, job))
    _q_ctr += 1


def _q_pop(wc):
    """Pop the highest-priority job from the WC heap queue. O(log n)."""
    _, _, job = heapq.heappop(wc.queue)
    return job


# Result accumulators
wo_completed = []
op_records   = []


def start_op_on_machine(wc, m, job, now):
    """Assign job to machine m and schedule EV_OP_DONE."""
    wc.mark_busy(m)
    finish = now + job["proc_days_cur"]
    push(finish, EV_OP_DONE, {
        "wc":          wc.name,
        "machine_id":  m.id,
        "job":         job,
        "start_time":  now,
        "finish_time": finish,
    })


def arrive_at_wc(wc, job, now):
    """Job arrives at a workcenter: unlimited -> fixed delay, limited -> queue or machine."""
    job["arrive_wc_time"] = now

    if wc.cap_type == "unlimited":
        delay  = config.FIXED_LT.get(wc.name, 1.0)
        finish = now + delay
        push(finish, EV_FIXED_DONE, {
            "wc":          wc.name,
            "job":         job,
            "start_time":  now,
            "finish_time": finish,
        })
    else:
        idle = wc.find_idle()
        if idle:
            job["queue_time_cur"] = 0.0
            start_op_on_machine(wc, idle, job, now)
        else:
            _q_push(wc, job)


def advance_job(job, now, workcenters, op_rec=None):
    """Move job to its next operation, or mark it complete."""
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

    t_arrive = next_workday_sim(now + config.TRANSPORT_DELAY_WD)

    wc = workcenters.get(wc_name)
    if wc is None:
        push(t_arrive + 1.0, EV_FIXED_DONE, {
            "wc":          wc_name,
            "job":         job,
            "start_time":  t_arrive,
            "finish_time": t_arrive + 1.0,
        })
    else:
        arrive_at_wc(wc, job, t_arrive)


def schedule_releases(df_wo, routing_dict):
    """
    Schedule EV_RELEASE events for all work orders.
    Uses itertuples (faster than iterrows).
    """
    released = 0
    for wo in df_wo.itertuples(index=False):
        if wo.item_name not in routing_dict:
            continue
        ops   = routing_dict[wo.item_name]
        rel_t = next_workday_sim(to_sim(wo.start_date.date()))
        job   = {
            "wo_num":      str(wo.wo_num),
            "item_name":   str(wo.item_name),
            "ord_qty":     int(wo.ord_qty),
            "due_sim":     to_sim(wo.due_date.date()),
            "release_sim": rel_t,
            "ops":         ops,
            "op_idx":      -1,
        }
        push(rel_t, EV_RELEASE, {"job": job})
        released += 1
    return released


def schedule_gfc_checks():
    """Schedule periodic GFC check events for the full simulation horizon."""
    t   = config.MONITOR_INTERVAL
    cnt = 0
    while t <= config.SIM_END_DAYS:
        push(t, EV_GFC_CHECK, {"t": t})
        t   += config.MONITOR_INTERVAL
        cnt += 1
    return cnt


def run_simulation(workcenters):
    """
    Main discrete-event loop.

    Processes all events in chronological order until SIM_END_DAYS.
    Returns number of processed events.
    """
    processed = 0

    # Pre-build list of limited WCs to avoid dict iteration on every event.
    limited_wcs = [wc for wc in workcenters.values() if wc.cap_type == "limited"]

    # Only sweep limited_wcs when time crosses the next snap boundary.
    _next_snap = config.SNAP_INTERVAL

    while _events:
        now, _, etype, payload = heapq.heappop(_events)
        if now > config.SIM_END_DAYS:
            break
        processed += 1

        if now >= _next_snap:
            for wc in limited_wcs:
                wc.record_snapshot(now)
            _next_snap = now + config.SNAP_INTERVAL

        if etype == EV_RELEASE:
            advance_job(payload["job"], now, workcenters)

        elif etype == EV_OP_DONE:
            wc_name = payload["wc"]
            wc      = workcenters[wc_name]
            m       = wc.machines[payload["machine_id"]]
            job     = payload["job"]
            t0      = payload["start_time"]

            wc.mark_idle(m)
            m.total_busy      += job["proc_days_cur"]
            wc.total_ops_done += 1
            q_time             = t0 - job["arrive_wc_time"]
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

            if wc.queue:
                nxt = _q_pop(wc)
                nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                start_op_on_machine(wc, m, nxt, now)

            advance_job(job, now, workcenters, rec)

        elif etype == EV_FIXED_DONE:
            job = payload["job"]
            rec = {
                "wo_num":      job["wo_num"],
                "item_name":   job["item_name"],
                "wc":          payload["wc"],
                "operation":   (
                    job["ops"][job["op_idx"]]["operation"]
                    if job["op_idx"] < len(job["ops"]) else -1
                ),
                "ord_qty":     job["ord_qty"],
                "proc_days":   payload["finish_time"] - payload["start_time"],
                "planned_q":   0.0,
                "arrive_wc":   payload["start_time"],
                "start_time":  payload["start_time"],
                "finish_time": payload["finish_time"],
                "queue_time":  0.0,
                "due_sim":     job["due_sim"],
            }
            advance_job(job, payload["finish_time"], workcenters, rec)

        elif etype == EV_GFC_CHECK:
            GFC.gfc_monitor(now, workcenters, push, EV_EXPAND_HOURS, EV_EXPAND_MACHINE)

        elif etype == EV_EXPAND_HOURS:
            wc    = workcenters[payload["wc"]]
            old_h = wc.hours_per_day
            new_h = payload["new_hours"]
            wc.hours_per_day      = new_h
            wc.tier1_effective_at = None
            print(f"  [SIM  {sim_to_date(now)}] Tier-1 effective: "
                  f"{wc.name} now running {new_h} h/day (was {old_h} h)")

        elif etype == EV_EXPAND_MACHINE:
            wc     = workcenters[payload["wc"]]
            second = payload.get("second", False)

            new_m = Machine(wc.n_machines, wc.name)
            wc.machines.append(new_m)
            wc._idle.append(new_m)   # new machine starts idle

            if second:
                print(f"  [SIM  {sim_to_date(now)}] Tier-2 (2nd machine) effective: "
                      f"{wc.name} +1 machine (now {wc.n_machines} total)")
            elif not wc.tier2_machine_added:
                wc.tier2_machine_added = True
                print(f"  [SIM  {sim_to_date(now)}] Tier-2 effective: "
                      f"{wc.name} +1 machine (now {wc.n_machines} total)")

            if wc.queue:
                nxt = _q_pop(wc)
                nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                start_op_on_machine(wc, new_m, nxt, now)

            # Ensure this WC is in the snapshot list (in case it wasn't limited before)
            if wc not in limited_wcs:
                limited_wcs.append(wc)

    return processed
