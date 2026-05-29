"""Query Decomposition：将复合问题拆解为可独立检索的原子子问题。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.state import AgentRoute

Aspect = Literal["impact", "handling", "cause", "manual", "log", "general"]

_ASPECT_IMPACT = re.compile(r"(是否影响|会不会影响|能否影响|对.*影响|运行影响|业务影响|有影响吗)")
_ASPECT_HANDLING = re.compile(
    r"(如何处理|怎么处理|怎么办|如何恢复|如何排查|如何处置|处置步骤|排查步骤|应如何|该怎么)"
)
_ASPECT_CAUSE = re.compile(r"(原因|为何|为什么|可能原因|故障原因|什么原因)")
_ASPECT_MANUAL = re.compile(
    r"(手册|参数|配置|功能|协议|接口|操作步骤|规范|定义|限制|支持哪些)"
)

_RE_IMPACT_THEN_HANDLING = re.compile(
    r"^(?P<prefix>.+?)"
    r"(?P<impact>(?:是否|会不会|能否)(?:.{0,12})?影响(?:业务|运行)?(?:[\u4e00-\u9fff]{0,8})?)"
    r"[，,；;]\s*"
    r"(?P<handling>(?:应)?(?:该)?(?:如何|怎么).+)[？?]?$"
)
_RE_CAUSE_THEN_HANDLING = re.compile(
    r"^(?P<prefix>.+?)"
    r"(?P<cause>(?:的)?(?:原因|为何|为什么).{0,30}?)"
    r"[，,；;]\s*"
    r"(?P<handling>(?:如何|怎么).+?)[？?]?$"
)
_RE_CONJUNCTION = re.compile(r"[，,；;]\s*(?=(?:如何|怎么|是否|能否|会不会|有没有))")

_LLM_DECOMPOSE_SYSTEM = (
    "你是空管设备运维知识库查询分解器。只输出一行合法 JSON，不要 Markdown，不要解释。"
)

_VALID_ASPECTS = frozenset({"impact", "handling", "cause", "manual", "log", "general"})


@dataclass
class SubQuery:
    text: str
    aspect: Aspect = "general"
    prefer_manual: bool = False
    prefer_log: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "aspect": self.aspect,
            "prefer_manual": self.prefer_manual,
            "prefer_log": self.prefer_log,
        }


@dataclass
class DecomposeResult:
    original_query: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    decomposer_type: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "decomposer_type": self.decomposer_type,
            "sub_query_texts": [sq.text for sq in self.sub_queries],
        }


def _llm_decompose_model() -> str:
    from rag.config import LLM_MODEL

    explicit = os.getenv("AGENT_LLM_DECOMPOSE_MODEL", "").strip()
    return explicit or LLM_MODEL


def _llm_decompose_temperature() -> float:
    raw = os.getenv("AGENT_LLM_DECOMPOSE_TEMPERATURE", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(2.0, float(raw)))
    except ValueError:
        return 0.0


def _max_sub_queries() -> int:
    raw = os.getenv("AGENT_DECOMPOSE_MAX_SUB_QUERIES", "4").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 4


def _llm_decompose_env_enabled() -> bool:
    v = os.getenv("AGENT_LLM_DECOMPOSE", "").strip().lower()
    return v in ("1", "true", "yes")


def _detect_aspect(text: str) -> Aspect:
    t = text.strip()
    if _ASPECT_IMPACT.search(t):
        return "impact"
    if _ASPECT_HANDLING.search(t):
        return "handling"
    if _ASPECT_CAUSE.search(t):
        return "cause"
    if _ASPECT_MANUAL.search(t):
        return "manual"
    if any(kw in t for kw in ("案例", "历史", "故障", "告警", "发生过")):
        return "log"
    return "general"


def _aspect_preferences(aspect: Aspect, route: AgentRoute) -> tuple[bool, bool]:
    if aspect == "impact":
        return False, True
    if aspect == "handling":
        if route == "manual_query":
            return True, False
        if route == "log_case_query":
            return False, True
        return True, True
    if aspect == "cause":
        return False, True
    if aspect == "manual":
        return True, False
    if aspect == "log":
        return False, True
    if route == "manual_query":
        return True, False
    if route == "log_case_query":
        return False, True
    if route in ("hybrid_diagnosis", "follow_up_query"):
        return True, True
    return False, False


def _make_sub_query(text: str, route: AgentRoute) -> SubQuery:
    q = text.strip().rstrip("？?").strip()
    aspect = _detect_aspect(q)
    prefer_manual, prefer_log = _aspect_preferences(aspect, route)
    return SubQuery(
        text=q,
        aspect=aspect,
        prefer_manual=prefer_manual,
        prefer_log=prefer_log,
    )


def _dedupe_sub_queries(subs: list[SubQuery]) -> list[SubQuery]:
    seen: set[str] = set()
    out: list[SubQuery] = []
    for sq in subs:
        key = sq.text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sq)
    return out


def _entity_prefix(clause: str) -> str:
    """从子句中抽取设备/主题前缀，供后续子句补全实体。"""
    text = clause.strip()
    m = re.match(
        r"^(.+?)(?:是否|会不会|能否|如何|怎么|为何|为什么|的?(?:原因|影响|处置|处理))",
        text,
    )
    if m:
        return m.group(1).strip().rstrip("，,；;的")
    return text


def _split_compound_clauses(q: str) -> list[str]:
    """按常见中文复合问句模式拆分为子句。"""
    text = q.strip()

    m = _RE_IMPACT_THEN_HANDLING.match(text)
    if m:
        prefix = _entity_prefix(m.group("prefix"))
        impact = m.group("impact").strip()
        handling = m.group("handling").strip()
        return [f"{prefix}{impact}", f"{prefix}{handling}"]

    m = _RE_CAUSE_THEN_HANDLING.match(text)
    if m:
        prefix = _entity_prefix(m.group("prefix"))
        cause = m.group("cause").strip()
        handling = m.group("handling").strip()
        return [f"{prefix}{cause}", f"{prefix}{handling}"]

    if re.search(r"[？?].+[？?]", text):
        parts = [p.strip() for p in re.split(r"[？?]+", text) if p.strip()]
        if len(parts) >= 2:
            prefix = parts[0]
            out = [parts[0]]
            for part in parts[1:]:
                if not re.search(r"(雷达|系统|设备|自动化|塔台|机坪|ADS-B|EFS|THALES|进程单)", part):
                    out.append(f"{_entity_prefix(parts[0])}{part}")
                else:
                    out.append(part)
            return out

    parts = _RE_CONJUNCTION.split(text)
    if len(parts) >= 2:
        aspects = {_detect_aspect(p) for p in parts}
        if len(aspects - {"general"}) >= 2:
            out: list[str] = [parts[0].strip()]
            entity = _entity_prefix(parts[0])
            for part in parts[1:]:
                p = part.strip()
                if p and not re.search(
                    r"(雷达|系统|设备|自动化|塔台|机坪|ADS-B|EFS|THALES|进程单)", p
                ):
                    out.append(f"{entity}{p}")
                else:
                    out.append(p)
            return out

    return [text]


def needs_decomposition(query: str, route: AgentRoute) -> bool:
    q = query.strip()
    if not q or route == "insufficient_evidence":
        return False
    if len(_split_compound_clauses(q)) >= 2:
        return True
    # 多意图关键词
    multi_kws = ("影响", "处理", "原因", "步骤", "案例", "手册", "恢复", "排查")
    hit = sum(1 for kw in multi_kws if kw in q)
    if hit >= 2 and (("，" in q or ";" in q or "；" in q) or len(q) > 28):
        return True
    if route in ("hybrid_diagnosis", "follow_up_query") and hit >= 2:
        return True
    return False


def rule_decompose_query(
    query: str,
    *,
    route: AgentRoute,
    needs_manual: bool = False,
    needs_log: bool = False,
) -> DecomposeResult:
    q = query.strip()
    if not q:
        return DecomposeResult(original_query=q, sub_queries=[], decomposer_type="rule")

    clauses = _split_compound_clauses(q)
    subs = [_make_sub_query(clause, route) for clause in clauses if clause.strip()]
    subs = _dedupe_sub_queries(subs)

    if not subs:
        subs = [_make_sub_query(q, route)]

    if len(subs) == 1 and needs_manual and needs_log and route in (
        "hybrid_diagnosis",
        "follow_up_query",
    ):
        base = subs[0]
        subs = [
            base,
            SubQuery(
                text=f"{base.text} 手册 操作步骤 参数",
                aspect="manual",
                prefer_manual=True,
                prefer_log=False,
            ),
            SubQuery(
                text=f"{base.text} 历史案例 处置 影响",
                aspect="log",
                prefer_manual=False,
                prefer_log=True,
            ),
        ]

    subs = subs[: _max_sub_queries()]
    return DecomposeResult(original_query=q, sub_queries=subs, decomposer_type="rule")


def extract_json_object(text: str) -> dict | None:
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


def build_llm_decompose_prompt(
    query: str,
    *,
    route: AgentRoute,
    needs_manual: bool,
    needs_log: bool,
    rule_result: DecomposeResult | None = None,
) -> str:
    hint = ""
    if rule_result and rule_result.sub_queries:
        hint = (
            "\n规则分解参考（可采纳或修正）："
            + json.dumps(rule_result.to_dict(), ensure_ascii=False)
        )
    return f"""请将用户问题拆解为若干「原子子问题」，每个子问题应可独立检索、只回答一个方面。

【要求】
- 只输出 JSON：{{"sub_queries":[{{"text":"...","aspect":"impact|handling|cause|manual|log|general","prefer_manual":false,"prefer_log":true}}]}}
- 每个 sub_query 须完整、可独立检索，保留设备/系统/故障等关键实体
- 复合问题（如「是否影响业务，如何处理」）必须拆成至少 2 条
- aspect 含义：impact=业务影响，handling=处置步骤，cause=原因，manual=手册规范，log=历史案例
- prefer_manual/prefer_log 指示优先检索源；双源排查时可两者均为 true
- 最多 {_max_sub_queries()} 条；简单单意图问题可只输出 1 条
- 不要 Markdown，不要解释

【路由】{route}（needs_manual={needs_manual}, needs_log={needs_log}）

【用户问题】
{query.strip()}
{hint}
"""


def _validate_llm_sub_queries(
    obj: dict,
    *,
    route: AgentRoute,
    original: str,
) -> list[SubQuery] | None:
    raw = obj.get("sub_queries")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[SubQuery] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(_make_sub_query(text, route))
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        aspect = str(item.get("aspect") or "general").strip()
        if aspect not in _VALID_ASPECTS:
            aspect = _detect_aspect(text)
        prefer_manual = bool(item.get("prefer_manual", False))
        prefer_log = bool(item.get("prefer_log", False))
        if not prefer_manual and not prefer_log:
            prefer_manual, prefer_log = _aspect_preferences(aspect, route)  # type: ignore[arg-type]
        out.append(
            SubQuery(
                text=text,
                aspect=aspect,  # type: ignore[arg-type]
                prefer_manual=prefer_manual,
                prefer_log=prefer_log,
            )
        )
    out = _dedupe_sub_queries(out)
    if not out:
        return None
    if len(out) == 1 and out[0].text == original and needs_decomposition(original, route):
        return None
    return out[: _max_sub_queries()]


def llm_decompose_query(
    query: str,
    *,
    route: AgentRoute,
    needs_manual: bool = False,
    needs_log: bool = False,
    rule_result: DecomposeResult | None = None,
) -> DecomposeResult | None:
    prompt = build_llm_decompose_prompt(
        query,
        route=route,
        needs_manual=needs_manual,
        needs_log=needs_log,
        rule_result=rule_result,
    )
    try:
        from rag.llm import chat

        text = chat(
            _LLM_DECOMPOSE_SYSTEM,
            prompt,
            model=_llm_decompose_model(),
            temperature=_llm_decompose_temperature(),
        )
    except Exception:
        return None

    obj = extract_json_object(text)
    if obj is None:
        return None
    subs = _validate_llm_sub_queries(obj, route=route, original=query.strip())
    if subs is None:
        return None
    return DecomposeResult(
        original_query=query.strip(),
        sub_queries=subs,
        decomposer_type="llm",
    )


def should_use_llm_decompose(
    query: str,
    route: AgentRoute,
    rule_result: DecomposeResult,
    *,
    use_llm_decompose: bool | None = None,
) -> bool:
    if use_llm_decompose is False:
        return False
    if use_llm_decompose is True:
        return True
    if not _llm_decompose_env_enabled():
        return False
    if route == "insufficient_evidence":
        return False
    if needs_decomposition(query, route) and len(rule_result.sub_queries) <= 1:
        return True
    if len(query.strip()) > 45:
        return True
    return False


def decompose_query(
    query: str,
    *,
    route: AgentRoute,
    needs_manual: bool = False,
    needs_log: bool = False,
    use_llm_decompose: bool | None = None,
) -> DecomposeResult:
    """标准 Query Decomposition 入口：规则拆解，必要时 LLM 增强。"""
    q = query.strip()
    if not q or route == "insufficient_evidence":
        return DecomposeResult(original_query=q, sub_queries=[], decomposer_type="none")

    rule_result = rule_decompose_query(
        q,
        route=route,
        needs_manual=needs_manual,
        needs_log=needs_log,
    )

    if should_use_llm_decompose(q, route, rule_result, use_llm_decompose=use_llm_decompose):
        llm_result = llm_decompose_query(
            q,
            route=route,
            needs_manual=needs_manual,
            needs_log=needs_log,
            rule_result=rule_result,
        )
        if llm_result is not None:
            return llm_result
        if use_llm_decompose is True:
            return DecomposeResult(
                original_query=q,
                sub_queries=rule_result.sub_queries,
                decomposer_type="rule_fallback",
            )

    return rule_result
