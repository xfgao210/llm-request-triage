"""Policies built directly on the canonical simulator scheduling contract."""

from llm_triage.simulation import (
    Action,
    CandidateView,
    Decision,
    ResourceView,
    SchedulerCallback,
    SchedulerSnapshot,
)
from .baselines import baseline_schedulers
from .fcfs import FCFSScheduler
from .length_aware import LengthAwareScheduler
from .slo_aware import StaticSLOAwareScheduler
from .burst_switch import BurstSwitchConfig, BurstSwitchScheduler

Scheduler = SchedulerCallback

__all__ = [
    "BurstSwitchConfig",
    "BurstSwitchScheduler",
    "Action",
    "CandidateView",
    "Decision",
    "FCFSScheduler",
    "LengthAwareScheduler",
    "ResourceView",
    "Scheduler",
    "SchedulerCallback",
    "SchedulerSnapshot",
    "StaticSLOAwareScheduler",
    "baseline_schedulers",
]
