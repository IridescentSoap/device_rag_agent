"""多轮上下文：追问检测与 standalone query 改写（规则版）。"""

from __future__ import annotations

import re
from typing import Any

# 依赖上一轮用户问题的追问表述
_FOLLOWUP_RE = re.compile(
    r"(它|这个|那个|这事|上述|刚才|前面|还会|影响吗|怎么办|怎么处理|如何处理|"
    r"需要重启吗|要重启吗|然后呢|还有吗|严重吗|能解决吗|怎么恢复|然后呢)",
    re.IGNORECASE,
)

# 过于模糊、缺少对象
_VAGUE_RE = re.compile(r"^(怎么办|怎么处理|还有吗|然后呢|严重吗)[？?]?$")


def _last_user_turn(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    for turn in reversed(history):
        role = (turn.get("role") or turn.get("type") or "").lower()
        if role in ("user", "human"):
            content = (turn.get("content") or turn.get("text") or "").strip()
            if content:
                return content
    return ""


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


def rewrite_query(
    query: str,
    history: list[dict[str, Any]] | None = None,
    *,
    use_llm: bool = False,
) -> str:
    """
    将追问改写为可独立检索的 query。
    use_llm=True 时预留 LLM 改写接口（当前仍走规则）。
    """
    q = query.strip()
    if not q:
        return q
    if use_llm:
        # 预留：可接入 rag.llm.chat 做 rewrite
        pass
    if not needs_history(q, history):
        return q
    prev = _last_user_turn(history)
    if not prev:
        return q
    return f"{prev}；补充追问：{q}"


def is_insufficient_query(query: str) -> bool:
    q = query.strip()
    if len(q) < 4:
        return True
    if _VAGUE_RE.match(q):
        return True
    return False
