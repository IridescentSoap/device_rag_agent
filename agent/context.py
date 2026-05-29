"""多轮上下文：追问检测与 standalone query 改写（规则 + LLM）。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# 依赖上一轮用户问题的追问表述
_FOLLOWUP_RE = re.compile(
    r"(它|这个|那个|这事|上述|刚才|前面|还会|影响吗|怎么办|怎么处理|如何处理|"
    r"需要重启吗|要重启吗|然后呢|还有吗|严重吗|能解决吗|怎么恢复|然后呢)",
    re.IGNORECASE,
)

# 过于模糊、缺少对象
_VAGUE_RE = re.compile(r"^(怎么办|怎么处理|还有吗|然后呢|严重吗)[？?]?$")

_LLM_REWRITE_SYSTEM = (
    "你是空管设备运维知识库查询改写器。"
    "将多轮对话中的追问改写为可独立检索的单条中文问句。"
    "需结合完整对话历史消解指代，包括较早轮次中的设备名、故障码与处置上下文。"
    "只输出改写后的问句，不要 Markdown，不要解释，不要编号。"
)


@dataclass
class RewriteResult:
    query: str
    rewriter_type: str = "none"  # none | rule | llm | rule_fallback

    def to_dict(self) -> dict[str, str]:
        return {"query": self.query, "rewriter_type": self.rewriter_type}


def _normalize_turn(turn: Any) -> tuple[str, str] | None:
    if isinstance(turn, dict):
        role = str(turn.get("role") or turn.get("type") or "user").lower()
        content = str(turn.get("content") or turn.get("text") or "").strip()
    else:
        role = "user"
        content = str(turn).strip()
    if not content:
        return None
    if role in ("human",):
        role = "user"
    if role in ("ai", "bot"):
        role = "assistant"
    return role, content


def _history_turns(history: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    if not history:
        return []
    out: list[tuple[str, str]] = []
    for turn in history:
        normalized = _normalize_turn(turn)
        if normalized:
            out.append(normalized)
    return out


def _last_user_turn(history: list[dict[str, Any]] | None) -> str:
    for role, content in reversed(_history_turns(history)):
        if role == "user":
            return content
    return ""


def _llm_rewrite_history_turns() -> int:
    raw = os.getenv("AGENT_LLM_REWRITE_HISTORY_TURNS", "").strip()
    if not raw:
        try:
            from rag.config import AGENT_LLM_REWRITE_HISTORY_TURNS

            return max(1, min(20, AGENT_LLM_REWRITE_HISTORY_TURNS))
        except Exception:
            return 5
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 5


def _format_history_turns(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return "（无）"
    lines = [f"{role}: {content}" for role, content in turns]
    return "\n".join(lines)


def format_history_for_llm_rewrite(
    history: list[dict[str, Any]] | None,
    *,
    max_turns: int | None = None,
) -> str:
    """LLM 改写用：最近 N 轮完整对话（含 user/assistant），默认 5 轮。"""
    window = max_turns if max_turns is not None else _llm_rewrite_history_turns()
    turns = _history_turns(history)[-window:]
    return _format_history_turns(turns)


def _format_history(history: list[dict[str, Any]] | None) -> str:
    """兼容旧调用：默认与 LLM 改写窗口一致。"""
    return format_history_for_llm_rewrite(history)


def needs_history(query: str, history: list[dict[str, Any]] | None) -> bool:
    q = query.strip()
    if not q or not history:
        return False
    if len(q) <= 12 and _FOLLOWUP_RE.search(q):
        return True
    if _VAGUE_RE.match(q):
        return True
    if _FOLLOWUP_RE.search(q) and len(q) < 40:
        return True
    return False


def _llm_rewrite_model() -> str:
    from rag.config import LLM_MODEL

    explicit = os.getenv("AGENT_LLM_REWRITE_MODEL", "").strip()
    return explicit or LLM_MODEL


def _llm_rewrite_temperature() -> float:
    raw = os.getenv("AGENT_LLM_REWRITE_TEMPERATURE", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(2.0, float(raw)))
    except ValueError:
        return 0.0


def _llm_rewrite_env_enabled() -> bool:
    v = os.getenv("AGENT_LLM_REWRITE", "").strip().lower()
    return v in ("1", "true", "yes")


def rule_rewrite_query(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> RewriteResult:
    q = query.strip()
    if not q:
        return RewriteResult(query=q, rewriter_type="none")
    if not needs_history(q, history):
        return RewriteResult(query=q, rewriter_type="none")
    prev = _last_user_turn(history)
    if not prev:
        return RewriteResult(query=q, rewriter_type="none")
    return RewriteResult(
        query=f"{prev}；补充追问：{q}",
        rewriter_type="rule",
    )


def build_llm_rewrite_prompt(
    query: str,
    history: list[dict[str, Any]] | None,
    *,
    rule_hint: str | None = None,
    history_turns: int | None = None,
) -> str:
    window = history_turns if history_turns is not None else _llm_rewrite_history_turns()
    prev = _last_user_turn(history)
    hint = ""
    if rule_hint:
        hint = f"\n规则改写参考（可采纳或修正）：{rule_hint}"
    return f"""请将「当前追问」结合对话历史改写为一条可独立检索的中文问句。

【要求】
- 阅读最近 {window} 轮完整对话（含 assistant 回复），消解指代（它/这个/上述/前面/那台等）
- 若关键实体出现在较早轮次，须从历史中找回并写入改写结果
- 补全设备名、故障现象、错误码、已尝试处置等关键信息
- 输出一条完整问句，可直接用于知识库检索
- 不要 Markdown，不要解释，不要输出多条
- 若追问本身已足够完整，可轻微润色但不要改变意图

【最近 {window} 轮对话历史】
{format_history_for_llm_rewrite(history, max_turns=window)}

【上一轮用户问题（快速参考）】
{prev or "（无）"}

【当前追问】
{query.strip()}
{hint}
"""


def _validate_llm_rewrite(text: str, *, original: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    s = s.strip().strip("\"'「」")
    s = re.sub(r"^改写[：:]\s*", "", s)
    s = s.split("\n")[0].strip()
    if not s or len(s) < 4:
        return None
    if len(s) > 300:
        return None
    if s == original.strip():
        return None
    return s


def llm_rewrite_query(
    query: str,
    history: list[dict[str, Any]] | None,
    *,
    rule_hint: str | None = None,
    history_turns: int | None = None,
) -> RewriteResult | None:
    prompt = build_llm_rewrite_prompt(
        query,
        history,
        rule_hint=rule_hint,
        history_turns=history_turns,
    )
    try:
        from rag.llm import chat

        text = chat(
            _LLM_REWRITE_SYSTEM,
            prompt,
            model=_llm_rewrite_model(),
            temperature=_llm_rewrite_temperature(),
        )
    except Exception:
        return None

    rewritten = _validate_llm_rewrite(text, original=query)
    if not rewritten:
        return None
    return RewriteResult(query=rewritten, rewriter_type="llm")


def should_use_llm_rewrite(
    query: str,
    history: list[dict[str, Any]] | None,
    *,
    use_llm_rewrite: bool | None = None,
) -> bool:
    if use_llm_rewrite is False:
        return False
    if use_llm_rewrite is True:
        return needs_history(query, history)
    if not _llm_rewrite_env_enabled():
        return False
    return needs_history(query, history)


def rewrite_query_with_meta(
    query: str,
    history: list[dict[str, Any]] | None = None,
    *,
    use_llm_rewrite: bool | None = None,
    use_llm: bool | None = None,  # 兼容旧参数名
) -> RewriteResult:
    """将追问改写为可独立检索的 query（规则优先，必要时 LLM 增强）。"""
    if use_llm_rewrite is None and use_llm is not None:
        use_llm_rewrite = use_llm

    q = query.strip()
    if not q:
        return RewriteResult(query=q, rewriter_type="none")

    rule_result = rule_rewrite_query(q, history)
    if not needs_history(q, history):
        return RewriteResult(query=q, rewriter_type="none")

    if not should_use_llm_rewrite(q, history, use_llm_rewrite=use_llm_rewrite):
        return rule_result

    llm_result = llm_rewrite_query(q, history, rule_hint=rule_result.query)
    if llm_result is not None:
        return llm_result

    if use_llm_rewrite is True:
        return RewriteResult(
            query=rule_result.query,
            rewriter_type="rule_fallback",
        )
    return rule_result


def rewrite_query(
    query: str,
    history: list[dict[str, Any]] | None = None,
    *,
    use_llm_rewrite: bool | None = None,
    use_llm: bool | None = None,
) -> str:
    return rewrite_query_with_meta(
        query,
        history,
        use_llm_rewrite=use_llm_rewrite,
        use_llm=use_llm,
    ).query


def is_insufficient_query(query: str) -> bool:
    q = query.strip()
    if len(q) < 4:
        return True
    if _VAGUE_RE.match(q):
        return True
    return False
