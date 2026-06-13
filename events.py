"""
events.py -- SDG Shopfloor Simulation
======================================
Discrete-event engine.

Contains:
  - Event-type constants
  - Event queue (min-heap via heapq)
  - Job routing helpers: arrive_at_wc, advance_job, start_op_on_machine
  - run_simulation(): main loop that processes all events up to SIM_END_DAYS

All side-effects (WO completion, operation records) are accumulated into
the lists `wo_completed` and `op_records`, which main.py converts to DataFrames.
"""

from __future__ import annotations
import heapq
from datetime import date, timedelta
from typing import TYPE_CHECKING

import config
from workcenter import WorkCenter, Machine
import gfc as GFC

if TYPE_CHECKING:
    pass


# Event-type constants
EV_RELEASE        = 0   # WO released to shopfloor -> start first operation
EV_OP_DONE        = 2   # Operation finished on a machine
EV_FIXED_DONE     = 3   # Fixed-LT step completed
EV_GFC_CHECK      = 4   # Periodic GFC monitoring
EV_EXPAND_HOURS   = 5   # Tier-1 expansion effective: update hours_per_day
EV_EXPAND_MACHINE = 6   # Tier-2 expansion effective: add machine object

# Event queue
_events: list = []
_ctr    = 0   # tie-breaker to keep heap ordering deterministic


def push(t: float, etype: int, payload: dict) -> None:
    """Insert a new event into the priority queue."""
    global _ctr
    heapq.heappush(_events, (t, _ctr, etype, payload))
    _ctr += 1


# Date helpers
def to_sim(d: date) -> float:
    return float((d - config.SIM_ORIGIN).days)


def sim_to_date(t: float) -> date:
    return config.SIM_ORIGIN + timedelta(days=int(t))


def next_workday_sim(t: float) -> float:
    """Advance to the next Monday if `t` falls on a weekend."""
    d    = config.SIM_ORIGIN + timedelta(days=int(t))
    frac = t - int(t)
    while d.weekday() >= 5:   # 5=Sat, 6=Sun
        d    += timedelta(days=1)
        frac  = 0.0
    return to_sim(d) + frac


# Dispatch key
def dispatch_key(job: dict) -> float:
    if config.DISPATCH_RULE == "EDD":
        return job["due_sim"]
    if config.DISPATCH_RULE == "SPT":
        return job["proc_days_cur"]
    return job["arrive_wc_time"]   # FIFO


# Result accumulators
wo_completed: list[dict] = []
op_records:   list[dict] = []


# Core routing functions

def start_op_on_machine(
    wc: WorkCenter, m: Machine, job: dict, now: float
) -> None:
    """Assign job to machine m and schedule EV_OP_DONE."""
    m.busy  = True
    finish  = now + job["proc_days_cur"]
    push(finish, EV_OP_DONE, {
        "wc":          wc.name,
        "machine_id":  m.id,
        "job":         job,
        "start_time":  now,
        "finish_time": finish,
    })


def arrive_at_wc(
    wc: WorkCenter, job: dict, now: float
) -> None:
    """
    Job arrives at a workcenter.
    - Unlimited-capacity WC: schedule fixed-delay completion immediately.
    - Limited-capacity WC: assign to idle machine or add to queue.
    """
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
            wc.queue.append(job)


def advance_job(
    job: dict,
    now: float,
    workcenters: dict[str, WorkCenter],
    op_rec: dict | None = None,
) -> None:
    """
    Move job to its next operation, or record it as complete if routing is done.
    """
    if op_rec:
        op_records.append(op_rec)

    ops    = job["ops"]
    op_idx = job["op_idx"] + 1
    job["op_idx"] = op_idx

    if op_idx >= len(ops):
        # All operations complete -> work order done
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

    # Prepare next operation
    op      = ops[op_idx]
    wc_name = op["wc"]
    proc    = op["setup_days"] + op["run_days"] * job["ord_qty"]
    job["proc_days_cur"] = proc
    job["planned_q_cur"] = op["planned_qtime"]
    job["cur_wc"]        = wc_name

    # Gate new start to next workday (transport delay included)
    t_arrive = next_workday_sim(now + config.TRANSPORT_DELAY_WD)

    wc = workcenters.get(wc_name)
    if wc is None:
        # Unknown WC -- treat as 1-day fixed delay (documented assumption)
        push(t_arrive + 1.0, EV_FIXED_DONE, {
            "wc":          wc_name,
            "job":         job,
            "start_time":  t_arrive,
            "finish_time": t_arrive + 1.0,
        })
    else:
        arrive_at_wc(wc, job, t_arrive)


# Simulation loop

def schedule_releases(
    df_wo,
    routing_dict: dict[str, list[dict]],
) -> int:
    """
    Schedule EV_RELEASE events for all work orders.

    Returns
    -------
    released : int -- number of WOs successfully scheduled
    """
    released = 0
    for _, wo in df_wo.iterrows():
        if wo["item_name"] not in routing_dict:
            continue
        ops   = routing_dict[wo["item_name"]]
        rel_t = next_workday_sim(to_sim(wo["start_date"].date()))
        job   = {
            "wo_num":      str(wo["wo_num"]),
            "item_name":   str(wo["item_name"]),
            "ord_qty":     int(wo["ord_qty"]),
            "due_sim":     to_sim(wo["due_date"].date()),
            "release_sim": rel_t,
            "ops":         ops,
            "op_idx":      -1,   # incremented to 0 on first advance_job call
        }
        push(rel_t, EV_RELEASE, {"job": job})
        released += 1
    return released


def schedule_gfc_checks() -> int:
    """Schedule periodic GFC check events for the full simulation horizon."""
    t   = config.MONITOR_INTERVAL
    cnt = 0
    while t <= config.SIM_END_DAYS:
        push(t, EV_GFC_CHECK, {"t": t})
        t   += config.MONITOR_INTERVAL
        cnt += 1
    return cnt


def run_simulation(workcenters: dict[str, WorkCenter]) -> int:
    """
    Main discrete-event loop.

    Processes all events in chronological order until SIM_END_DAYS.
    Modifies workcenters in place and appends to wo_completed / op_records.

    Returns
    -------
    processed_events : int
    """
    processed = 0

    while _events:
        now, _, etype, payload = heapq.heappop(_events)
        if now > config.SIM_END_DAYS:
            break
        processed += 1

        # Snapshots for every limited WC
        for wc in workcenters.values():
            if wc.cap_type == "limited":
                wc.record_snapshot(now)

        # WO RELEASE
        if etype == EV_RELEASE:
            advance_job(payload["job"], now, workcenters)

        # OPERATION DONE
        elif etype == EV_OP_DONE:
            wc_name = payload["wc"]
            wc      = workcenters[wc_name]
            m       = wc.machines[payload["machine_id"]]
            job     = payload["job"]
            t0      = payload["start_time"]

            m.busy         = False
            m.total_busy  += job["proc_days_cur"]
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

            # Dispatch next queued job to the now-free machine
            if wc.queue:
                wc.queue.sort(key=dispatch_key)
                nxt = wc.queue.pop(0)
                nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                start_op_on_machine(wc, m, nxt, now)

            advance_job(job, now, workcenters, rec)

        # FIXED-LT DONE
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

        # GFC PERIODIC CHECK
        elif etype == EV_GFC_CHECK:
            GFC.gfc_monitor(
                now, workcenters, push,
                EV_EXPAND_HOURS, EV_EXPAND_MACHINE,
            )

        # TIER-1: HOURS EXTENSION EFFECTIVE
        elif etype == EV_EXPAND_HOURS:
            wc      = workcenters[payload["wc"]]
            old_h   = wc.hours_per_day
            new_h   = payload["new_hours"]
            wc.hours_per_day      = new_h
            wc.tier1_effective_at = None   # extension is now live; allow next step
            print(
                f"  [SIM  {sim_to_date(now)}] Tier-1 effective: "
                f"{wc.name} now running {new_h} h/day (was {old_h} h)"
            )

        # TIER-2: NEW MACHINE ONLINE
        elif etype == EV_EXPAND_MACHINE:
            wc     = workcenters[payload["wc"]]
            second = payload.get("second", False)

            if second:
                # WC_BW exception: second extra machine
                new_m = Machine(wc.n_machines, wc.name)
                wc.machines.append(new_m)
                print(
                    f"  [SIM  {sim_to_date(now)}] Tier-2 (2nd machine) effective: "
                    f"{wc.name} +1 machine (now {wc.n_machines} total)"
                )
                if wc.queue:
                    wc.queue.sort(key=dispatch_key)
                    nxt = wc.queue.pop(0)
                    nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                    start_op_on_machine(wc, new_m, nxt, now)

            elif not wc.tier2_machine_added:
                # Normal first extra machine (any WC)
                new_m = Machine(wc.n_machines, wc.name)
                wc.machines.append(new_m)
                wc.tier2_machine_added = True
                print(
                    f"  [SIM  {sim_to_date(now)}] Tier-2 effective: "
                    f"{wc.name} +1 machine (now {wc.n_machines} total)"
                )
                if wc.queue:
                    wc.queue.sort(key=dispatch_key)
                    nxt = wc.queue.pop(0)
                    nxt["queue_time_cur"] = now - nxt["arrive_wc_time"]
                    start_op_on_machine(wc, new_m, nxt, now)

    return processed
