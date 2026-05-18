"""Query 路由 + 分源 TopK + BM25/向量双路 + RRF 融合。"""

from __future__ import annotations

import math
import re
from typing import Literal

import numpy as np
from qdrant_client import QdrantClient

from rag.config import (
    LOG_DEDUP_MAX_PER_SYSTEM,
    RRF_K,
    ROUTE_KEYWORD_HINT,
    ROUTE_RATIO,
    ROUTE_TOPK_SUM,
    ROUTE_USE_CORPUS_NORM,
    TOPK_BM25,
    TOPK_PER_SOURCE,
    TOPK_VECTOR,
)
from rag.embedder import BgeEmbedder
from rag.index_store import CorpusIndex, DualIndexStore
from rag.schemas import ChunkRecord

QueryRoute = Literal["manual_heavy", "log_heavy", "balanced"]

# 轻量关键词：辅助路由，仅在语义分数接近时生效或加权
_LOG_ROUTE_KWS = (
    "故障",
    "告警",
    "现象",
    "处置",
    "恢复",
    "重启",
    "案例",
    "历史",
    "停机",
    "黑屏",
    "卡死",
    "宕",
    "备件",
    "更换",
    "影响",
    "无影响",
    "运行影响",
    "可能原因",
    "怎么办",
    "如何恢复",
)
_LOG_CASE_RE = re.compile(
    r"处置|恢复|案例|故障|告警|黑屏|卡死|宕机|宕掉|备件|更换|怎么办|可能原因|运行影响|无影响|如何恢复|历史"
)
_MANUAL_ROUTE_KWS = (
    "手册",
    "功能",
    "参数",
    "配置",
    "流程",
    "协议",
    "支持哪些",
    "如何描述",
    "定义",
    "限制",
    "条件",
    "ASTERIX",
    "接口",
    "格式",
    "章节",
    "DEP",
    "CDN",
    "移交",
    "降级",
    "告警功能",
    "探测",
)


def _top_sum_rrf(scores: dict[str, float], k: int) -> float:
    if not scores:
        return 0.0
    vals = sorted(scores.values(), reverse=True)
    kk = min(k, len(vals))
    return float(sum(vals[:kk]))


def _mean_top_rrf(scores: dict[str, float], k: int) -> float:
    if not scores:
        return 0.0
    vals = sorted(scores.values(), reverse=True)
    kk = min(k, len(vals))
    return float(sum(vals[:kk]) / kk)


def _corpus_norm_factor(n_chunks: int, ref: int = 500) -> float:
    """小语料（手册）略增益，大语料（日志）略衰减，缓解规模失衡。"""
    if n_chunks <= 0:
        return 1.0
    return math.sqrt(math.log1p(ref) / math.log1p(n_chunks))


def is_log_case_query(query: str) -> bool:
    """运维案例/故障处置类问法，应避免判为 manual_heavy。"""
    if _LOG_CASE_RE.search(query):
        return True
    log_hits = sum(1 for kw in _LOG_ROUTE_KWS if kw in query)
    manual_spec = sum(
        1
        for kw in _MANUAL_ROUTE_KWS
        if kw in query and kw not in ("告警", "降级", "移交")
    )
    return log_hits >= 1 and manual_spec == 0


def query_route_keyword_hint(query: str) -> QueryRoute | None:
    """返回 None 表示无显著关键词倾向。"""
    if not ROUTE_KEYWORD_HINT:
        return None
    if is_log_case_query(query):
        return "log_heavy"
    log_hits = sum(1 for kw in _LOG_ROUTE_KWS if kw in query)
    manual_hits = sum(1 for kw in _MANUAL_ROUTE_KWS if kw in query)
    if log_hits >= 2 and log_hits > manual_hits:
        return "log_heavy"
    if manual_hits >= 2 and manual_hits > log_hits:
        return "manual_heavy"
    if manual_hits >= 1 and log_hits == 0 and re.search(
        r"功能|参数|手册|协议|格式|条件|支持哪些|如何描述", query
    ):
        return "manual_heavy"
    return None


def classify_query_route(
    manual_scores: dict[str, float],
    log_scores: dict[str, float],
    *,
    topk_sum: int = ROUTE_TOPK_SUM,
    ratio: float = ROUTE_RATIO,
    manual_n: int = 0,
    log_n: int = 0,
    query: str = "",
) -> QueryRoute:
    """
    路由：Top-K RRF 均值（抗规模差异）+ 可选语料归一化 + 关键词轻量加权/打破平局。
    """
    s_m = _mean_top_rrf(manual_scores, topk_sum)
    s_l = _mean_top_rrf(log_scores, topk_sum)

    if ROUTE_USE_CORPUS_NORM:
        if manual_n > 0:
            s_m *= _corpus_norm_factor(manual_n)
        if log_n > 0:
            s_l *= _corpus_norm_factor(log_n)

    hint = query_route_keyword_hint(query) if query else None
    if hint == "log_heavy":
        s_l *= 1.12
    elif hint == "manual_heavy":
        s_m *= 1.12

    if s_m <= 0 and s_l <= 0:
        return hint or "balanced"
    if s_m <= 0:
        return "log_heavy"
    if s_l <= 0:
        return "manual_heavy"

    if s_m >= ratio * s_l:
        route: QueryRoute = "manual_heavy"
    elif s_l >= ratio * s_m:
        route = "log_heavy"
    else:
        route = "balanced"

    # 语义接近时，关键词倾向可打破 balanced
    if route == "balanced" and hint in ("manual_heavy", "log_heavy"):
        if max(s_m, s_l) > 0 and min(s_m, s_l) / max(s_m, s_l) >= 0.75:
            return hint

    # P0.1：案例/故障类 query 禁止 manual_heavy（如「罕山雷达…处置」）
    if query and is_log_case_query(query):
        if route == "manual_heavy":
            if s_l > 0 and s_l >= s_m * 0.7:
                return "log_heavy"
            return "balanced"
        if route == "balanced" and hint == "log_heavy":
            return "log_heavy"

    if query and hint == "log_heavy" and route == "manual_heavy":
        return "log_heavy" if s_l > 0 else "balanced"

    return route


def _rrf_merge(
    lists: list[list[ChunkRecord]],
    k: int = RRF_K,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _topk_from_scores(
    chunks: list[ChunkRecord],
    scores: np.ndarray,
    k: int,
) -> list[ChunkRecord]:
    if not chunks or scores.size == 0:
        return []
    n = min(k, len(chunks))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [chunks[int(i)] for i in idx]


def retrieve_from_corpus(
    corpus: CorpusIndex,
    query: str,
    embedder: BgeEmbedder,
    qdrant_client: QdrantClient,
    topk_bm25: int = TOPK_BM25,
    topk_vec: int = TOPK_VECTOR,
) -> dict[str, float]:
    if not corpus.chunks:
        return {}
    bm = corpus.bm25_scores(query)
    ranked_b = _topk_from_scores(corpus.chunks, bm, topk_bm25)
    ranked_v = corpus.vector_search(query, embedder, qdrant_client, topk_vec)
    return _rrf_merge([ranked_b, ranked_v])


def chunks_from_corpus_scores(
    corpus: CorpusIndex,
    scores: dict[str, float],
    top_n: int,
) -> list[ChunkRecord]:
    if not scores or not corpus.chunks:
        return []
    id_map = {c.chunk_id: c for c in corpus.chunks}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [id_map[cid] for cid, _ in ranked if cid in id_map]


def dedupe_log_chunks(
    chunks: list[ChunkRecord],
    max_per_system: int = LOG_DEDUP_MAX_PER_SYSTEM,
) -> list[ChunkRecord]:
    """同系统过多相似 case 时只保留前列，降低 THALES/EFS 等刷屏。"""
    if max_per_system <= 0:
        return chunks
    counts: dict[str, int] = {}
    out: list[ChunkRecord] = []
    for c in chunks:
        key = (c.meta.get("system") or "").strip() or c.text[:48]
        n = counts.get(key, 0)
        if n >= max_per_system:
            continue
        counts[key] = n + 1
        out.append(c)
    return out


def merge_dual_rrf(
    manual_scores: dict[str, float],
    log_scores: dict[str, float],
    route: QueryRoute,
) -> dict[str, float]:
    w_m, w_l = 1.0, 1.0
    if route == "manual_heavy":
        w_m, w_l = 1.5, 0.65
    elif route == "log_heavy":
        w_m, w_l = 0.65, 1.5
    out: dict[str, float] = {}
    for cid, s in manual_scores.items():
        out[cid] = out.get(cid, 0.0) + w_m * s
    for cid, s in log_scores.items():
        out[cid] = out.get(cid, 0.0) + w_l * s
    return out


def scores_to_chunks(
    store: DualIndexStore,
    merged: dict[str, float],
    top_n: int,
) -> list[ChunkRecord]:
    id_map: dict[str, ChunkRecord] = {}
    for c in store.manual.chunks:
        id_map[c.chunk_id] = c
    for c in store.log.chunks:
        id_map[c.chunk_id] = c
    ranked = sorted(merged.items(), key=lambda x: -x[1])[:top_n]
    return [id_map[cid] for cid, s in ranked if cid in id_map]


def rerank_quota_for_route(route: QueryRoute, rerank_topk: int) -> tuple[int, int]:
    """返回 (manual_rerank_n, log_rerank_n)。"""
    if route == "manual_heavy":
        return max(8, rerank_topk * 3 // 4), max(4, rerank_topk // 6)
    if route == "log_heavy":
        return max(4, rerank_topk // 6), max(8, rerank_topk * 3 // 4)
    half = max(6, rerank_topk // 2)
    return half, half


def merge_quota_for_route(
    route: QueryRoute,
    final_topk: int,
    *,
    cross_max: int = 1,
) -> tuple[int, int]:
    """返回 (max_manual, max_log) 进入最终列表的条数上限。"""
    if route == "log_heavy":
        return cross_max, final_topk
    if route == "manual_heavy":
        return final_topk, cross_max
    half = max(3, final_topk // 2)
    return half + cross_max, half + cross_max
