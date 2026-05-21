"""Query Planner：规则意图识别 + LLM Planner（可选）。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace

from agent.context import is_insufficient_query, needs_history
from agent.state import AgentRoute, PlanResult

VALID_PLANNER_ROUTES: frozenset[str] = frozenset(
    {
        "manual_query",
        "log_case_query",
        "hybrid_diagnosis",
        "follow_up_query",
        "insufficient_evidence",
    }
)

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

_LLM_PLANNER_SYSTEM = (
    "你是空管设备运维知识库查询规划器。只输出一行合法 JSON，不要 Markdown，不要解释。"
)


def _llm_planner_model() -> str:
    from rag.config import LLM_MODEL

    explicit = os.getenv("AGENT_LLM_PLANNER_MODEL", "").strip()
    return explicit or LLM_MODEL


def _llm_planner_temperature() -> float:
    raw = os.getenv("AGENT_LLM_PLANNER_TEMPERATURE", "").strip()
    if not raw:
        return 0.1
    try:
        return max(0.0, min(2.0, float(raw)))
    except ValueError:
        return 0.1

_MULTI_INTENT_KWS = ("影响", "处理", "原因", "步骤", "案例", "手册")


def _count_kws(text: str, kws: tuple[str, ...]) -> int:
    return sum(1 for kw in kws if kw in text)


def _route_needs(route: str) -> tuple[bool, bool]:
    if route == "manual_query":
        return True, False
    if route == "log_case_query":
        return False, True
    if route in ("hybrid_diagnosis", "follow_up_query"):
        return True, True
    return False, False


def _format_history(history: list | None) -> str:
    if not history:
        return "（无）"
    lines: list[str] = []
    for turn in history[-6:]:
        if isinstance(turn, dict):
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "").strip()
        else:
            role = "user"
            content = str(turn).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "（无）"


def build_llm_planner_prompt(
    query: str,
    history: list | None = None,
    rewritten_query: str | None = None,
    rule_plan: PlanResult | None = None,
) -> str:
    q = (rewritten_query or query).strip()
    rule_hint = ""
    if rule_plan is not None:
        rule_hint = (
            f"\n规则 Planner 参考（可采纳或修正）：route={rule_plan.route}, "
            f"confidence={rule_plan.confidence}, "
            f"needs_manual={rule_plan.needs_manual}, needs_log={rule_plan.needs_log}"
        )

    return f"""请为以下用户问题生成查询规划，输出严格 JSON（不要 Markdown，不要解释文字）。

【角色】空管设备运维知识库查询规划器。

【可选 route】
- manual_query：查设备手册、参数、功能、操作步骤、配置说明
- log_case_query：查历史故障、处置案例、是否影响业务、恢复方式
- hybrid_diagnosis：同时需要查手册和日志的综合排查问题
- follow_up_query：依赖历史上下文的追问
- insufficient_evidence：问题过于模糊，需用户补充设备/系统名、故障现象或手册主题

【规划要求】
- 只能输出 JSON，字段如下：
{{
  "route": "...",
  "rewritten_query": "...",
  "sub_queries": ["..."],
  "needs_manual": true,
  "needs_log": true,
  "missing_info": ["..."],
  "confidence": 0.0
}}
- 不要输出 Markdown，不要输出解释文字
- 问题模糊时 route 用 insufficient_evidence，missing_info 列出需补充项
- 同时涉及「是否影响业务」和「如何处理」时，优先 hybrid_diagnosis
- 若是追问，结合 history 改写 rewritten_query
- needs_manual / needs_log 须与 route 一致（manual 仅手册，log 仅日志，hybrid/follow_up 双源）

【用户问题】
{query.strip()}

【改写后 query（上下文改写结果，可在此基础上再优化）】
{q}

【对话历史】
{_format_history(history)}
{rule_hint}
"""


def extract_json_object(text: str) -> dict | None:
    """从纯 JSON 或 ```json ... ``` 代码块中提取对象。"""
    s = (text or "").strip()
    if not s:
        return None

    block = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if block:
        s = block.group(1).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def validate_plan_dict(obj: dict, rule_plan: PlanResult) -> PlanResult | None:
    route = str(obj.get("route") or "").strip()
    if route not in VALID_PLANNER_ROUTES:
        return None

    rewritten = str(obj.get("rewritten_query") or "").strip()
    if not rewritten:
        rewritten = rule_plan.rewritten_query

    raw_sub = obj.get("sub_queries")
    if isinstance(raw_sub, list) and all(isinstance(x, str) for x in raw_sub):
        sub_queries = [x.strip() for x in raw_sub if x.strip()]
    else:
        sub_queries = []
    if not sub_queries:
        sub_queries = [rewritten]
    if route == "insufficient_evidence":
        sub_queries = []

    raw_missing = obj.get("missing_info")
    if isinstance(raw_missing, list) and all(isinstance(x, str) for x in raw_missing):
        missing_info = [x.strip() for x in raw_missing if x.strip()]
    else:
        missing_info = []

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.55:
        return None

    needs_manual, needs_log = _route_needs(route)

    return PlanResult(
        route=route,  # type: ignore[arg-type]
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        needs_manual=needs_manual,
        needs_log=needs_log,
        confidence=confidence,
        planner_type="llm",
        missing_info=missing_info,
    )


def llm_plan_query(
    query: str,
    history: list | None = None,
    *,
    rewritten_query: str | None = None,
    rule_plan: PlanResult | None = None,
) -> PlanResult | None:
    rp = rule_plan or rule_plan_query(
        query, history, rewritten_query=rewritten_query
    )
    user_prompt = build_llm_planner_prompt(
        query, history, rewritten_query=rewritten_query, rule_plan=rp
    )
    try:
        from rag.llm import chat

        text = chat(
            _LLM_PLANNER_SYSTEM,
            user_prompt,
            model=_llm_planner_model(),
            temperature=_llm_planner_temperature(),
        )
    except Exception:
        return None

    obj = extract_json_object(text)
    if obj is None:
        return None
    return validate_plan_dict(obj, rp)


def rule_plan_query(
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
            planner_type="rule",
            missing_info=[],
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
        planner_type="rule",
        missing_info=[],
    )


def _llm_planner_env_enabled() -> bool:
    v = os.getenv("AGENT_LLM_PLANNER", "").strip().lower()
    return v in ("1", "true", "yes")


def _count_multi_intent(query: str) -> int:
    return sum(1 for kw in _MULTI_INTENT_KWS if kw in query)


def should_use_llm_planner(
    query: str,
    history: list | None,
    rule_plan: PlanResult,
    use_llm_planner: bool | None = None,
) -> bool:
    if use_llm_planner is False:
        return False
    if use_llm_planner is True:
        return True

    if not _llm_planner_env_enabled():
        return False

    raw = query.strip()
    if rule_plan.route == "hybrid_diagnosis":
        return True
    if rule_plan.route == "follow_up_query":
        return True
    if rule_plan.confidence < 0.8:
        return True
    if len(raw) > 40:
        return True
    if _count_multi_intent(raw) >= 2:
        return True
    if history:
        return True
    return False


def _as_rule_fallback(rule_plan: PlanResult) -> PlanResult:
    return replace(rule_plan, planner_type="rule_fallback")


def plan_query(
    query: str,
    history: list | None = None,
    *,
    rewritten_query: str | None = None,
    use_llm_planner: bool | None = None,
) -> PlanResult:
    rule_plan = rule_plan_query(query, history, rewritten_query=rewritten_query)
    if not should_use_llm_planner(query, history, rule_plan, use_llm_planner):
        return rule_plan

    try:
        llm_plan = llm_plan_query(
            query,
            history,
            rewritten_query=rewritten_query,
            rule_plan=rule_plan,
        )
    except Exception:
        llm_plan = None

    if llm_plan is not None:
        return llm_plan
    return _as_rule_fallback(rule_plan)
