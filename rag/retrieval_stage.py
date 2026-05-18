"""单语料检索阶段：BM25 + 向量 -> RRF -> Rerank -> 阈值过滤。"""

from __future__ import annotations

from qdrant_client import QdrantClient

from rag.config import RERANK_MODEL, TOPK_BM25, TOPK_VECTOR
from rag.embedder import BgeEmbedder
from rag.index_store import CorpusIndex
from rag.rerank import RerankHit, rerank_chunks_with_scores
from rag.retrieve import retrieve_from_corpus


def corpus_retrieve_rerank(
    corpus: CorpusIndex,
    query: str,
    embedder: BgeEmbedder,
    qdrant_client: QdrantClient,
    *,
    topk_bm25: int = TOPK_BM25,
    topk_vec: int = TOPK_VECTOR,
    pool_size: int = 100,
    rerank_topk: int = 50,
    threshold: float = 0.7,
    final_topk: int = 20,
    fill_shortage: bool = False,
) -> list[RerankHit]:
    merged_scores = retrieve_from_corpus(
        corpus,
        query,
        embedder,
        qdrant_client,
        topk_bm25=topk_bm25,
        topk_vec=topk_vec,
    )
    if not merged_scores:
        return []

    id_map = {c.chunk_id: c for c in corpus.chunks}
    rrf_ranked = sorted(merged_scores.items(), key=lambda x: -x[1])[:pool_size]
    pool = [id_map[cid] for cid, _ in rrf_ranked if cid in id_map]
    if not pool:
        return []

    rerank_hits = rerank_chunks_with_scores(
        query,
        pool,
        top_n=min(rerank_topk, len(pool)),
        model=RERANK_MODEL,
    )
    passed = [h for h in rerank_hits if h.score >= threshold][:final_topk]
    if fill_shortage and len(passed) < final_topk:
        selected_ids = {h.chunk.chunk_id for h in passed}
        for h in rerank_hits:
            if h.chunk.chunk_id in selected_ids:
                continue
            passed.append(h)
            selected_ids.add(h.chunk.chunk_id)
            if len(passed) >= final_topk:
                break
    return passed
