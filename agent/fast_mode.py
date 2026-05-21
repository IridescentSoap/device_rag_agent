"""Agent 快速模式：缩小候选池与 Rerank 规模，缩短单次查询耗时。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def resolve_fast_mode(explicit: bool | None = None) -> bool:
    """CLI/API 显式传参优先，否则读 AGENT_FAST_MODE。"""
    if explicit is not None:
        return explicit
    return _env_bool("AGENT_FAST_MODE", "0")


@dataclass(frozen=True)
class FastRetrieveSettings:
    per_source_pool: int
    rerank_topk: int
    final_topk: int
    manual_pool_size: int
    manual_top_k: int
    log_pool_size: int
    log_top_k: int
    context_n: int


def get_fast_settings() -> FastRetrieveSettings:
    return FastRetrieveSettings(
        per_source_pool=int(os.environ.get("AGENT_FAST_PER_SOURCE_POOL", "24")),
        rerank_topk=int(os.environ.get("AGENT_FAST_RERANK_TOPK", "20")),
        final_topk=int(os.environ.get("AGENT_FAST_FINAL_TOPK", "12")),
        manual_pool_size=int(os.environ.get("AGENT_FAST_MANUAL_POOL", "24")),
        manual_top_k=int(os.environ.get("AGENT_FAST_MANUAL_TOPK", "8")),
        log_pool_size=int(os.environ.get("AGENT_FAST_LOG_POOL", "24")),
        log_top_k=int(os.environ.get("AGENT_FAST_LOG_TOPK", "8")),
        context_n=int(os.environ.get("AGENT_FAST_CONTEXT_N", "6")),
    )


# Rerank 运行时环境覆盖（由 rag/rerank.py 读取）
_FAST_RERANK_ENV_KEYS = (
    "RAG_QWEN3_RERANK_MAX_LENGTH",
    "RAG_LOCAL_RERANK_BATCH_SIZE",
    "RAG_RERANK_DOC_MAX_CHARS",
)


@contextmanager
def fast_rerank_env() -> Iterator[None]:
    """在快速模式下临时收紧 Rerank 序列长度、batch 与文档截断。"""
    overrides = {
        "RAG_QWEN3_RERANK_MAX_LENGTH": os.environ.get(
            "AGENT_FAST_QWEN3_RERANK_MAX_LENGTH", "1024"
        ),
        "RAG_LOCAL_RERANK_BATCH_SIZE": os.environ.get(
            "AGENT_FAST_RERANK_BATCH_SIZE", "4"
        ),
        "RAG_RERANK_DOC_MAX_CHARS": os.environ.get(
            "AGENT_FAST_RERANK_DOC_CHARS", "1500"
        ),
    }
    saved: dict[str, str | None] = {}
    for key in _FAST_RERANK_ENV_KEYS:
        saved[key] = os.environ.get(key)
        os.environ[key] = overrides[key]
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def fast_mode_summary() -> dict[str, Any]:
    s = get_fast_settings()
    return {
        "enabled": True,
        "per_source_pool": s.per_source_pool,
        "rerank_topk": s.rerank_topk,
        "final_topk": s.final_topk,
        "qwen3_max_length": os.environ.get("AGENT_FAST_QWEN3_RERANK_MAX_LENGTH", "1024"),
        "rerank_batch_size": os.environ.get("AGENT_FAST_RERANK_BATCH_SIZE", "4"),
        "rerank_doc_max_chars": os.environ.get("AGENT_FAST_RERANK_DOC_CHARS", "1500"),
    }
