"""
workcenter.py — SDG Shopfloor Simulation
=========================================
Machine and WorkCenter classes with utilisation tracking and GFC expansion state.
"""

from __future__ import annotations
import config


class Machine:
    """
    A single machine resource inside a workcenter.

    Attributes
    ----------
    id          : int   — sequential index within the workcenter
    wc          : str   — parent workcenter name
    busy        : bool  — True while processing a job
    total_busy  : float — accumulated processing time (working days)
    """

    def __init__(self, mid: int, wc: str):
        self.id         = mid
        self.wc         = wc
        self.busy       = False
        self.total_busy = 0.0


class WorkCenter:
    """
    A workcenter consisting of one or more identical machines.

    Handles:
      - Job queuing and machine assignment
      - Utilisation tracking (instantaneous + rolling average)
      - GFC capacity-expansion state (Tier-1 hours, Tier-2 machine)

    Parameters
    ----------
    name          : workcenter identifier (e.g. "WC_AMILL_HS")
    n_machines    : initial machine count (0 for unlimited-capacity WCs)
    dept          : "AUTO" or "MANUAL"
    hours_per_day : available hours per machine per day (AUTO=21, MANUAL=16)
    cap_type      : "limited" | "unlimited"
    """

    def __init__(
        self,
        name: str,
        n_machines: int,
        dept: str,
        hours_per_day: float,
        cap_type: str,
    ):
        self.name          = name
        self.dept          = dept
        self.cap_type      = cap_type
        self.hours_per_day = hours_per_day   # current (may increase via Tier-1)
        self.base_hours    = hours_per_day   # original value (for reporting)

        self.machines: list[Machine] = (
            [Machine(i, name) for i in range(n_machines)] if n_machines else []
        )
        self.queue: list[dict] = []

        # ── Statistics ────────────────────────────────────────────────────────
        self.total_ops_done  = 0
        self.queue_time_sum  = 0.0
        # Time-series snapshots: list of (sim_t, value)
        self.queue_snaps: list[tuple[float, int]]   = []
        self.util_snaps:  list[tuple[float, float]] = []   # (t, busy_fraction 0-1)
        # WIP = jobs in queue + jobs currently on a machine
        self.wip_snaps:   list[tuple[float, int]]   = []   # (t, wip_count)

        # ── GFC capacity-expansion state ──────────────────────────────────────
        self.warning_active      = False              # currently in warning zone (70-85%)
        self.alert_active        = False              # currently in alert zone (>85%)
        self.warning_since: float | None = None       # sim_t when entered warning zone
        self.alert_since:   float | None = None       # sim_t when entered alert zone
        self.tier1_effective_at: float | None = None  # pending Tier-1 event time
        self.tier2_ordered       = False              # new machine already ordered
        self.tier2_effective_at: float | None = None
        self.tier2_machine_added = False
        # WC_BW exception: a second extra machine may be added if still overloaded
        # after the first Tier-2 machine is online and WC is already at 21 h/day.
        self.tier2_second_ordered = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_machines(self) -> int:
        return len(self.machines)

    # ── Machine assignment ────────────────────────────────────────────────────

    def find_idle(self) -> Machine | None:
        """Return the first idle machine, or None if all are busy."""
        for m in self.machines:
            if not m.busy:
                return m
        return None

    # ── Utilisation helpers ───────────────────────────────────────────────────

    def instantaneous_util(self) -> float:
        """Fraction of machines currently busy (0–1)."""
        if not self.machines:
            return 0.0
        return sum(1 for m in self.machines if m.busy) / len(self.machines)

    def rolling_util(
        self,
        now: float,
        window: float = config.ROLLING_WINDOW_DAYS,
    ) -> float:
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

    # ── Snapshot recording ────────────────────────────────────────────────────

    def record_snapshot(self, now: float):
        """
        Record instantaneous utilisation, queue-length, and WIP snapshots.
        Called by the main simulation loop at SNAP_INTERVAL cadence.

        WIP = jobs waiting in queue + jobs currently being processed on a machine.
        """
        if (
            not self.util_snaps
            or (now - self.util_snaps[-1][0]) >= config.SNAP_INTERVAL
        ):
            self.util_snaps.append((now, self.instantaneous_util()))

        if (
            not self.queue_snaps
            or (now - self.queue_snaps[-1][0]) >= config.SNAP_INTERVAL
        ):
            self.queue_snaps.append((now, len(self.queue)))

        if (
            not self.wip_snaps
            or (now - self.wip_snaps[-1][0]) >= config.SNAP_INTERVAL
        ):
            busy_count = sum(1 for m in self.machines if m.busy)
            self.wip_snaps.append((now, len(self.queue) + busy_count))


def build_workcenters() -> dict[str, WorkCenter]:
    """
    Instantiate all workcenters from WC_CONFIG.

    Returns
    -------
    dict mapping wc_name → WorkCenter instance.
    """
    wcs: dict[str, WorkCenter] = {}
    for wc_name, (nm, dept, hpd, cap) in config.WC_CONFIG.items():
        wcs[wc_name] = WorkCenter(
            name=wc_name,
            n_machines=nm if nm else 0,
            dept=dept,
            hours_per_day=hpd,
            cap_type=cap,
        )
    return wcs
