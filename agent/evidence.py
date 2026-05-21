"""证据充分性简单判断。"""

from __future__ import annotations

from agent.state import EvidenceItem, EvidenceResult
from rag.rerank import RerankHit


def _preview(text: str, n: int = 160) -> str:
    t = text.replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")


def judge_evidence(
    hits: list[RerankHit],
    *,
    pipeline_route: str = "balanced",
    plan_confidence: float = 0.8,
) -> EvidenceResult:
    if not hits:
        return EvidenceResult(
            items=[],
            citations=[],
            confidence=0.0,
            need_human_confirm=True,
            missing_aspects=["未召回到相关手册或日志片段"],
            pipeline_route=pipeline_route,
        )

    items: list[EvidenceItem] = []
    for h in hits:
        c = h.chunk
        items.append(
            EvidenceItem(
                chunk_id=c.chunk_id,
                source=c.source,
                score=float(h.score),
                text_preview=_preview(c.text),
                case_id=str(c.meta.get("case_id") or ""),
                doc_id=str(c.meta.get("doc_id") or ""),
            )
        )

    citations = [it.chunk_id for it in items]
    avg_score = sum(it.score for it in items) / len(items)
    confidence = min(0.95, max(0.4, avg_score * 0.85 + plan_confidence * 0.15))

    missing: list[str] = []
    if not any(it.source == "manual" for it in items):
        missing.append("缺少手册依据")
    if not any(it.source == "log" for it in items):
        missing.append("缺少历史案例依据")

    need_confirm = len(citations) == 0 or confidence < 0.45

    return EvidenceResult(
        items=items,
        citations=citations,
        confidence=round(confidence, 4),
        need_human_confirm=need_confirm,
        missing_aspects=missing,
        pipeline_route=pipeline_route,
    )
