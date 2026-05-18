"""空管设备运维 RAG（与 rag_plan_v2.md 对齐）。"""

from __future__ import annotations

__all__ = ["RagPipeline"]


def __getattr__(name: str):
    if name == "RagPipeline":
        from rag.pipeline import RagPipeline

        return RagPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
