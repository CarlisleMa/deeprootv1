"""Abstract base class for all agents.

Every agent (Phase 1 and Phase 2) inherits from BaseAgent, which provides:
- A GraphClient for Neo4j access
- A standard run() interface
- Logging and progress tracking
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.graph.client import GraphClient


class BaseAgent(ABC):
    """Base class for all agents in the pipeline."""

    def __init__(self, client: GraphClient, **kwargs: Any):
        self.client = client
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def run(self, **kwargs: Any) -> dict:
        """Execute the agent's task.

        Returns a summary dict with at minimum:
            {"nodes_created": int, "edges_created": int, "errors": list[str]}
        """
        ...

    def _log_progress(self, msg: str) -> None:
        self.logger.info(f"[{self.name}] {msg}")
