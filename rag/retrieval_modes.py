"""单路 / 双路（RRF）检索封装，便于评估对比。"""

from __future__ import annotations

from qdrant_client import QdrantClient

from rag.embedder import BgeEmbedder
from rag.index_store import CorpusIndex
from rag.retrieve import _topk_from_scores, retrieve_from_corpus


def ranked_bm25_only(
    corpus: CorpusIndex,
    query: str,
    topk: int,
) -> list[str]:
    if not corpus.chunks:
        return []
    scores = corpus.bm25_scores(query)
    ranked = _topk_from_scores(corpus.chunks, scores, topk)
    return [c.chunk_id for c in ranked]


def ranked_vector_only(
    corpus: CorpusIndex,
    query: str,
    embedder: BgeEmbedder,
    client: QdrantClient,
    topk: int,
) -> list[str]:
    chunks = corpus.vector_search(query, embedder, client, topk)
    return [c.chunk_id for c in chunks]


def ranked_rrf_single_corpus(
    corpus: CorpusIndex,
    query: str,
    embedder: BgeEmbedder,
    client: QdrantClient,
    topk_bm25: int,
    topk_vec: int,
    final_k: int,
) -> list[str]:
    """与线上单源一致：BM25 TopK + 向量 TopK → RRF 融合后按分数排序取前 final_k。"""
    scores = retrieve_from_corpus(
        corpus, query, embedder, client, topk_bm25, topk_vec
    )
    if not scores:
        return []
    merged = sorted(scores.items(), key=lambda x: -x[1])[:final_k]
    return [cid for cid, _ in merged]


def recall_at_k(gold_ids: set[str], ranked_ids: list[str], k: int) -> float:
    top = set(ranked_ids[:k])
    return 1.0 if (gold_ids & top) else 0.0


def mrr(gold_ids: set[str], ranked_ids: list[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0
