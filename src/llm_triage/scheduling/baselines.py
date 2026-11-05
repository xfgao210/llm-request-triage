"""Canonical deterministic policy set for baseline comparisons."""

from __future__ import annotations

from llm_triage.simulation import SchedulerCallback
from .fcfs import FCFSScheduler
from .length_aware import LengthAwareScheduler
from .slo_aware import StaticSLOAwareScheduler


def baseline_schedulers() -> dict[str, SchedulerCallback]:
    """Return a fresh, deterministically ordered baseline policy mapping."""

    return {
        "fcfs": FCFSScheduler(),
        "length_aware": LengthAwareScheduler(),
        "static_slo_aware": StaticSLOAwareScheduler(),
    }
