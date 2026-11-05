"""First-come, first-served baseline exposed under its historical name."""

from llm_triage.simulation import FCFSCallback


FCFSScheduler = FCFSCallback

__all__ = ["FCFSScheduler"]
