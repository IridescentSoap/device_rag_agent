"""Query Planner：规则意图识别。"""

from __future__ import annotations

import re

from agent.context import is_insufficient_query, needs_history
from agent.state import AgentRoute, PlanResult

_LOG_KWS = (
    "故障", "告警", "现象", "处置", "恢复", "重启", "案例", "历史", "黑屏", "卡死",
    "宕机", "备件", "更换", "影响", "无影响", "运行影响", "可能原因", "怎么办",
    "如何恢复", "是否影响业务", "有没有类似", "发生过",
)
_MANUAL_KWS = (
    "手册", "功能", "参数", "配置", "流程", "协议", "接口", "支持哪些", "如何描述",
    "定义", "限制", "ASTERIX", "章节", "DEP", "CDN", "移交", "降级", "探测", "QNH",
    "操作步骤", "说明", "规范",
)
_HYBRID_KWS = (
    "排查", "诊断", "综合", "同时", "结合手册", "参考手册", "怎么处理", "如何处理",
    "原因和", "处置和", "是否影响", "应该如何",
)
_DEVICE_KWS = re.compile(
    r"(雷达|自动化|进程单|塔台|机坪|ADS-B|EFS|THALES|QNH|RVSM|MTCA|系统)",
)


def _count_kws(text: str, kws: tuple[str, ...]) -> int:
    return sum(1 for kw in kws if kw in text)


def plan_query(
    query: str,
    history: list | None = None,
    *,
    rewritten_query: str | None = None,
) -> PlanResult:
    q = (rewritten_query or query).strip()
    raw = query.strip()

    if is_insufficient_query(raw) and not needs_history(raw, history):
        return PlanResult(
            route="insufficient_evidence",
            rewritten_query=q,
            sub_queries=[],
            needs_manual=False,
            needs_log=False,
            confidence=0.3,
        )

    if needs_history(raw, history):
        route: AgentRoute = "follow_up_query"
    else:
        log_h = _count_kws(q, _LOG_KWS)
        manual_h = _count_kws(q, _MANUAL_KWS)
        hybrid_h = _count_kws(q, _HYBRID_KWS)
        has_device = bool(_DEVICE_KWS.search(q))

        if hybrid_h >= 1 or (log_h >= 1 and manual_h >= 1):
            route = "hybrid_diagnosis"
        elif log_h >= 1 and (log_h >= manual_h or "影响" in q or "故障" in q):
            route = "log_case_query"
        elif manual_h >= 1:
            route = "manual_query"
        elif has_device and ("影响" in q or "处置" in q or "黑屏" in q):
            route = "hybrid_diagnosis"
        elif has_device:
            route = "log_case_query" if "?" in q or "吗" in q else "manual_query"
        else:
            route = "hybrid_diagnosis" if has_device else "insufficient_evidence"

    needs_manual = route in ("manual_query", "hybrid_diagnosis", "follow_up_query")
    needs_log = route in ("log_case_query", "hybrid_diagnosis", "follow_up_query")
    if route == "insufficient_evidence":
        needs_manual = needs_log = False

    sub: list[str] = [q]
    if needs_manual and needs_log:
        sub = [q, f"手册：{q}", f"日志案例：{q}"]
    elif needs_manual:
        sub = [q]
    elif needs_log:
        sub = [q]

    conf = 0.85
    if route == "insufficient_evidence":
        conf = 0.35
    elif route == "follow_up_query":
        conf = 0.75

    return PlanResult(
        route=route,
        rewritten_query=q,
        sub_queries=sub,
        needs_manual=needs_manual,
        needs_log=needs_log,
        confidence=conf,
    )
