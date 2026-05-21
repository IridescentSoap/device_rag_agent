"""证据充分性简单判断。"""

from __future__ import annotations

from agent.state import EvidenceItem, EvidenceResult
from rag.rerank import RerankHit

_IMPACT_QUERY_KWS = ("影响", "业务", "运行影响", "是否影响")
_IMPACT_EVIDENCE_KWS = ("影响", "无影响", "业务", "运行", "恢复", "中断")

_HANDLING_QUERY_KWS = ("怎么处理", "如何处理", "处置", "恢复", "排查", "怎么办")
_HANDLING_EVIDENCE_KWS = ("处置", "处理", "恢复", "重启", "检查", "排查", "更换", "解决")

_FAULT_QUERY_KWS = ("故障", "告警", "异常", "黑屏", "卡死", "无法启动")
_FAULT_EVIDENCE_KWS = ("故障", "告警", "异常", "原因", "现象", "可能原因")

_SUPPLEMENT_TEMPLATES: dict[str, str] = {
    "缺少手册依据": "{query} 手册 操作步骤 参数 说明",
    "缺少历史案例依据": "{query} 历史案例 处置 恢复 影响",
    "缺少业务影响说明": "{query} 是否影响业务 运行影响 无影响",
    "缺少处置步骤": "{query} 处置步骤 排查 恢复 重启",
    "缺少故障现象或原因说明": "{query} 故障现象 可能原因 告警",
}


def _preview(text: str, n: int = 160) -> str:
    t = text.replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")


def _contains_any(text: str, kws: tuple[str, ...]) -> bool:
    return any(kw in text for kw in kws)


def generate_supplement_queries(
    query: str | None,
    missing_aspects: list[str],
    route: str | None,
    max_queries: int = 3,
) -> list[str]:
    """根据缺失维度生成补充检索 query（去重，最多 max_queries 条）。"""
    if not query or not missing_aspects:
        return []

    q = query.strip()
    seen: set[str] = set()
    out: list[str] = []
    for aspect in missing_aspects:
        tpl = _SUPPLEMENT_TEMPLATES.get(aspect)
        if not tpl:
            continue
        candidate = tpl.format(query=q)
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= max_queries:
            break
    return out


def judge_evidence(
    hits: list[RerankHit],
    *,
    pipeline_route: str = "balanced",
    plan_confidence: float = 0.8,
    query: str | None = None,
    route: str | None = None,
) -> EvidenceResult:
    q = (query or "").strip()

    if not hits:
        missing = ["未召回到相关手册或日志片段"]
        supplements = generate_supplement_queries(q or None, missing, route)
        return EvidenceResult(
            items=[],
            citations=[],
            confidence=0.0,
            need_human_confirm=True,
            missing_aspects=missing,
            pipeline_route=pipeline_route,
            supplement_queries=supplements,
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
    evidence_text = " ".join(
        (h.chunk.text for h in hits if getattr(h, "chunk", None) is not None)
    )

    missing: list[str] = []
    check_dual_source = route == "hybrid_diagnosis" or pipeline_route == "balanced"
    if check_dual_source:
        if not any(it.source == "manual" for it in items):
            missing.append("缺少手册依据")
        if not any(it.source == "log" for it in items):
            missing.append("缺少历史案例依据")

    if q:
        if _contains_any(q, _IMPACT_QUERY_KWS) and not _contains_any(
            evidence_text, _IMPACT_EVIDENCE_KWS
        ):
            missing.append("缺少业务影响说明")
        if _contains_any(q, _HANDLING_QUERY_KWS) and not _contains_any(
            evidence_text, _HANDLING_EVIDENCE_KWS
        ):
            missing.append("缺少处置步骤")
        if _contains_any(q, _FAULT_QUERY_KWS) and not _contains_any(
            evidence_text, _FAULT_EVIDENCE_KWS
        ):
            missing.append("缺少故障现象或原因说明")

    avg_score = sum(it.score for it in items) / len(items)
    base_confidence = avg_score * 0.85 + plan_confidence * 0.15
    confidence = max(0.0, min(0.95, base_confidence - 0.08 * len(missing)))
    need_confirm = bool(missing) or confidence < 0.55

    supplements = generate_supplement_queries(q or None, missing, route)

    return EvidenceResult(
        items=items,
        citations=citations,
        confidence=round(confidence, 4),
        need_human_confirm=need_confirm,
        missing_aspects=missing,
        pipeline_route=pipeline_route,
        supplement_queries=supplements,
    )
