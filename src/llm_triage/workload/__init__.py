"""Trace ingestion, chronological windows, and synthetic workloads."""

from .synthetic import SyntheticConfig, generate_synthetic_workload

__all__ = ["SyntheticConfig", "generate_synthetic_workload"]
