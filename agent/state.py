"""Agent 运行状态与结构化输出。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentRoute = Literal[
    "manual_query",
    "log_case_query",
    "hybrid_diagnosis",
    "follow_up_query",
    "insufficient_evidence",
]

PipelineRoute = Literal["manual_heavy", "log_heavy", "balanced"]


@dataclass
class PlanResult:
    route: AgentRoute
    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    needs_manual: bool = False
    needs_log: bool = False
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "rewritten_query": self.rewritten_query,
            "sub_queries": self.sub_queries,
            "needs_manual": self.needs_manual,
            "needs_log": self.needs_log,
            "confidence": self.confidence,
        }


@dataclass
class EvidenceItem:
    chunk_id: str
    source: str
    score: float
    text_preview: str = ""
    case_id: str = ""
    doc_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": self.score,
            "text_preview": self.text_preview,
            "case_id": self.case_id,
            "doc_id": self.doc_id,
        }


@dataclass
class EvidenceResult:
    items: list[EvidenceItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    need_human_confirm: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    pipeline_route: str = "balanced"

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": self.citations,
            "confidence": self.confidence,
            "need_human_confirm": self.need_human_confirm,
            "missing_aspects": self.missing_aspects,
            "pipeline_route": self.pipeline_route,
            "evidence_count": len(self.items),
            "items": [x.to_dict() for x in self.items[:8]],
        }


@dataclass
class AgentResponse:
    answer: str
    route: str
    rewritten_query: str
    tools_used: list[str]
    citations: list[str]
    confidence: float
    need_human_confirm: bool
    latency_ms: int
    fast_mode: bool = False
    plan: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "answer": self.answer,
            "route": self.route,
            "rewritten_query": self.rewritten_query,
            "tools_used": self.tools_used,
            "citations": self.citations,
            "confidence": self.confidence,
            "need_human_confirm": self.need_human_confirm,
            "latency_ms": self.latency_ms,
            "fast_mode": self.fast_mode,
            "plan": self.plan,
            "evidence": self.evidence,
        }
        return out
