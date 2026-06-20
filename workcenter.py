"""
workcenter.py -- SDG Shopfloor Simulation
==========================================
Machine and WorkCenter classes with utilisation tracking and GFC expansion state.

Optimisations vs original:
  - find_idle() is O(1) via a deque of idle machines instead of O(n) linear scan.
  - _n_busy counter avoids recomputing sum(m.busy) in util helpers and snapshots.
  - rolling_util uses bisect for O(log n) cutoff lookup instead of O(n) list filter.
  - record_snapshot uses a single time-gate check for all three series.
"""

from __future__ import annotations
import bisect
from collections import deque
import config


class Machine:
    def __init__(self, mid: int, wc: str):
        self.id         = mid
        self.wc         = wc
        self.busy       = False
        self.total_busy = 0.0


class WorkCenter:
    """
    A workcenter consisting of one or more identical machines.

    Parameters
    ----------
    name          : workcenter identifier (e.g. "WC_AMILL_HS")
    n_machines    : initial machine count (0 for unlimited-capacity WCs)
    dept          : "AUTO" or "MANUAL"
    hours_per_day : available hours per machine per day (AUTO=21, MANUAL=16)
    cap_type      : "limited" | "unlimited"
    """

    def __init__(self, name, n_machines, dept, hours_per_day, cap_type):
        self.name          = name
        self.dept          = dept
        self.cap_type      = cap_type
        self.hours_per_day = hours_per_day
        self.base_hours    = hours_per_day

        self.machines = [Machine(i, name) for i in range(n_machines)] if n_machines else []

        # queue is a heapq: items are (priority_key, arrival_counter, job).
        # len(self.queue) still gives correct queue depth.
        self.queue = []

        # O(1) idle machine lookup via deque.
        # Invariant: _idle contains exactly the machines with busy=False.
        self._idle = deque(self.machines)

        # Busy counter -- avoids O(n) scan in util helpers.
        self._n_busy = 0

        # Statistics
        self.total_ops_done  = 0
        self.queue_time_sum  = 0.0
        self.queue_snaps = []   # list of (sim_t, queue_len)
        self.util_snaps  = []   # list of (sim_t, busy_fraction 0-1)
        self.wip_snaps   = []   # list of (sim_t, wip_count)

        # GFC capacity-expansion state
        self.warning_active       = False
        self.alert_active         = False
        self.warning_since        = None
        self.alert_since          = None
        self.tier1_effective_at   = None
        self.tier2_ordered        = False
        self.tier2_effective_at   = None
        self.tier2_machine_added  = False
        self.tier2_second_ordered = False

    @property
    def n_machines(self):
        return len(self.machines)

    # -- Machine assignment ----------------------------------------------------

    def find_idle(self):
        """Return an idle machine in O(1) via _idle deque, or None if all busy."""
        return self._idle[0] if self._idle else None

    def mark_busy(self, m):
        """Mark machine m as busy and remove it from the idle pool."""
        m.busy = True
        self._idle.remove(m)
        self._n_busy += 1

    def mark_idle(self, m):
        """Mark machine m as idle and return it to the idle pool."""
        m.busy = False
        self._idle.append(m)
        self._n_busy -= 1

    # -- Utilisation helpers ---------------------------------------------------

    def instantaneous_util(self):
        """Fraction of machines currently busy (0-1). O(1) via _n_busy counter."""
        if not self.machines:
            return 0.0
        return self._n_busy / len(self.machines)

    def rolling_util(self, now, window=None):
        """
        Rolling average utilisation (0-100%) over the last `window` sim-days.
        Uses bisect on sorted timestamps for O(log n) cutoff lookup.
        """
        if window is None:
            window = config.ROLLING_WINDOW_DAYS
        if not self.util_snaps:
            return 0.0
        cutoff = now - window
        keys = [t for t, _ in self.util_snaps]
        idx  = bisect.bisect_left(keys, cutoff)
        if idx >= len(self.util_snaps):
            idx = len(self.util_snaps) - 1
        recent = self.util_snaps[idx:]
        return 100.0 * sum(u for _, u in recent) / len(recent)

    # -- Snapshot recording ----------------------------------------------------

    def record_snapshot(self, now):
        """
        Record utilisation, queue-length, and WIP snapshots.
        Single time-gate check shared by all three series.
        """
        last_t = self.util_snaps[-1][0] if self.util_snaps else -1e9
        if (now - last_t) < config.SNAP_INTERVAL:
            return
        q_len = len(self.queue)
        util  = self._n_busy / len(self.machines) if self.machines else 0.0
        self.util_snaps.append((now, util))
        self.queue_snaps.append((now, q_len))
        self.wip_snaps.append((now, q_len + self._n_busy))


def build_workcenters():
    """Instantiate all workcenters from WC_CONFIG."""
    wcs = {}
    for wc_name, (nm, dept, hpd, cap) in config.WC_CONFIG.items():
        wcs[wc_name] = WorkCenter(
            name=wc_name,
            n_machines=nm if nm else 0,
            dept=dept,
            hours_per_day=hpd,
            cap_type=cap,
        )
    return wcs
