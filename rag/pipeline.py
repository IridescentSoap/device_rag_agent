"""端到端 RAG：检索 → 重排 → 上下文扩展 → 生成。"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import (
    FINAL_CONTEXT_N,
    INDEX_DIR,
    MERGE_MAX_CROSS_SOURCE,
    MIN_CONTEXT_N,
    PER_SOURCE_POOL,
    PER_SOURCE_RERANK,
    RERANK_TOP_N,
    TOPK_BM25,
    TOPK_PER_SOURCE,
    TOPK_VECTOR,
)
from rag.embedder import BgeEmbedder
from rag.index_store import DualIndexStore
from rag.llm import chat
from rag.prompts import build_user_message, log_chunk_body_for_prompt, system_prompt_for_route
from rag.rerank import RerankHit, rerank_chunks, rerank_chunks_with_scores
from rag.retrieve import (
    QueryRoute,
    chunks_from_corpus_scores,
    classify_query_route,
    dedupe_log_chunks,
    merge_dual_rrf,
    merge_quota_for_route,
    rerank_quota_for_route,
    retrieve_from_corpus,
    scores_to_chunks,
)
from rag.schemas import ChunkRecord


@dataclass
class RagAnswer:
    answer: str
    query_route: str
    citations: list[str]


@dataclass
class DualRetrieveResult:
    """双源检索 + 路由中间结果。"""

    query_route: QueryRoute
    manual_rrf_pool: int
    log_rrf_pool: int
    merged_pool: int
    recall_hits: list[RerankHit] = field(default_factory=list)


@dataclass
class DualRagAnswer(RagAnswer):
    """双源全流程回答（含召回明细）。"""

    recall_hits: list[RerankHit] = field(default_factory=list)
    manual_in_context: int = 0
    log_in_context: int = 0


def _neighbor_text(store: DualIndexStore, chunk: ChunkRecord) -> str:
    if chunk.source != "manual":
        return ""
    doc_id = chunk.meta.get("doc_id")
    idx = chunk.meta.get("chunk_index")
    if doc_id is None or idx is None:
        return ""
    extra: list[str] = []
    for delta in (-1, 1):
        want = idx + delta
        for c in store.manual.chunks:
            if c.meta.get("doc_id") == doc_id and c.meta.get("chunk_index") == want:
                extra.append(c.text[:2000])
                break
    return "\n".join(extra)


def expand_context(store: DualIndexStore, chunks: list[ChunkRecord]) -> list[str]:
    blocks: list[str] = []
    for c in chunks:
        header = f"[{c.source}] chunk_id={c.chunk_id}"
        if c.meta.get("case_id"):
            header += f" case_id={c.meta.get('case_id')}"
        if c.meta.get("page_range"):
            header += f" page={c.meta.get('page_range')}"
        neighbor = _neighbor_text(store, c)
        if c.source == "log":
            body = log_chunk_body_for_prompt(c)
        else:
            body = c.text
        if neighbor:
            body = f"{body}\n\n[相邻片段补充]\n{neighbor}"
        blocks.append(f"{header}\n{body}")
    return blocks


class RagPipeline:
    def __init__(
        self,
        index_dir: str | None = None,
        embedding_model: str | None = None,
        device: str | None = None,
    ):
        from rag import config

        self.index_dir = index_dir or str(INDEX_DIR)
        self.store = DualIndexStore.load(self.index_dir)
        model_name = embedding_model or config.EMBEDDING_MODEL
        self.embedder = BgeEmbedder(model_name, device=device)

    def retrieve(self, query: str, pool_size: int = 40) -> tuple[str, list[ChunkRecord]]:
        m_scores = retrieve_from_corpus(
            self.store.manual,
            query,
            self.embedder,
            self.store.client,
            TOPK_PER_SOURCE,
            TOPK_PER_SOURCE,
        )
        l_scores = retrieve_from_corpus(
            self.store.log,
            query,
            self.embedder,
            self.store.client,
            TOPK_PER_SOURCE,
            TOPK_PER_SOURCE,
        )
        route = classify_query_route(
            m_scores,
            l_scores,
            manual_n=len(self.store.manual.chunks),
            log_n=len(self.store.log.chunks),
            query=query,
        )
        merged = merge_dual_rrf(m_scores, l_scores, route)
        items = scores_to_chunks(self.store, merged, top_n=pool_size)
        return route, items

    def ask(self, query: str) -> RagAnswer:
        route, pool = self.retrieve(query)
        ranked = rerank_chunks(query, pool, top_n=max(RERANK_TOP_N, MIN_CONTEXT_N))
        take = ranked[:FINAL_CONTEXT_N] if len(ranked) >= MIN_CONTEXT_N else ranked
        ctx_blocks = expand_context(self.store, take)
        user_msg = build_user_message(query, ctx_blocks)
        answer = chat(system_prompt_for_route(route), user_msg)
        cites = [c.chunk_id for c in take]
        return RagAnswer(answer=answer, query_route=route, citations=cites)


class DualSourceRagPipeline:
    """
    双源全流程：manual/log 分源召回 -> query 路由 -> 加权 RRF 融合
    -> Rerank 阈值过滤 -> 按路由选 prompt -> LLM 生成。
    """

    def __init__(
        self,
        index_dir: str | None = None,
        embedding_model: str | None = None,
        device: str | None = None,
    ):
        from rag import config

        self.index_dir = index_dir or str(INDEX_DIR)
        self.store = DualIndexStore.load(self.index_dir)
        model_name = embedding_model or config.EMBEDDING_MODEL
        self.embedder = BgeEmbedder(model_name, device=device)

    def retrieve(
        self,
        query: str,
        *,
        topk_bm25: int = TOPK_BM25,
        topk_vec: int = TOPK_VECTOR,
        per_source_pool: int = 100,
        merge_pool_size: int = 80,
        rerank_topk: int = 50,
        threshold: float = 0.7,
        final_topk: int = 20,
        fill_shortage: bool = True,
    ) -> DualRetrieveResult:
        m_scores = retrieve_from_corpus(
            self.store.manual,
            query,
            self.embedder,
            self.store.client,
            topk_bm25=topk_bm25,
            topk_vec=topk_vec,
        )
        l_scores = retrieve_from_corpus(
            self.store.log,
            query,
            self.embedder,
            self.store.client,
            topk_bm25=topk_bm25,
            topk_vec=topk_vec,
        )
        n_manual = len(self.store.manual.chunks)
        n_log = len(self.store.log.chunks)
        route = classify_query_route(
            m_scores,
            l_scores,
            manual_n=n_manual,
            log_n=n_log,
            query=query,
        )

        if PER_SOURCE_RERANK:
            passed = self._retrieve_per_source_rerank(
                query,
                m_scores,
                l_scores,
                route,
                per_source_pool=PER_SOURCE_POOL,
                rerank_topk=rerank_topk,
                threshold=threshold,
                final_topk=final_topk,
                fill_shortage=fill_shortage,
            )
            merged_pool = len(passed)
        else:
            merged = merge_dual_rrf(m_scores, l_scores, route)
            pool = scores_to_chunks(self.store, merged, top_n=merge_pool_size)
            merged_pool = len(pool)
            if not pool:
                return DualRetrieveResult(
                    query_route=route,
                    manual_rrf_pool=len(m_scores),
                    log_rrf_pool=len(l_scores),
                    merged_pool=0,
                )
            rerank_hits = rerank_chunks_with_scores(
                query,
                pool,
                top_n=min(rerank_topk, len(pool)),
            )
            passed = self._filter_rerank_hits(
                rerank_hits, threshold, final_topk, fill_shortage
            )

        if not passed:
            return DualRetrieveResult(
                query_route=route,
                manual_rrf_pool=len(m_scores),
                log_rrf_pool=len(l_scores),
                merged_pool=0,
            )

        return DualRetrieveResult(
            query_route=route,
            manual_rrf_pool=len(m_scores),
            log_rrf_pool=len(l_scores),
            merged_pool=merged_pool,
            recall_hits=passed,
        )

    @staticmethod
    def _filter_rerank_hits(
        rerank_hits: list,
        threshold: float,
        final_topk: int,
        fill_shortage: bool,
    ) -> list:
        passed = [h for h in rerank_hits if h.score >= threshold][:final_topk]
        if fill_shortage and len(passed) < final_topk:
            selected = {h.chunk.chunk_id for h in passed}
            for h in rerank_hits:
                if h.chunk.chunk_id in selected:
                    continue
                passed.append(h)
                selected.add(h.chunk.chunk_id)
                if len(passed) >= final_topk:
                    break
        return passed

    def _retrieve_per_source_rerank(
        self,
        query: str,
        m_scores: dict[str, float],
        l_scores: dict[str, float],
        route: QueryRoute,
        *,
        per_source_pool: int,
        rerank_topk: int,
        threshold: float,
        final_topk: int,
        fill_shortage: bool,
    ) -> list:
        """分源候选池 + 分源 rerank + 按路由配额合并。"""
        manual_pool = chunks_from_corpus_scores(
            self.store.manual, m_scores, per_source_pool
        )
        log_pool = dedupe_log_chunks(
            chunks_from_corpus_scores(self.store.log, l_scores, per_source_pool)
        )
        m_quota, l_quota = rerank_quota_for_route(route, rerank_topk)

        all_hits: list = []
        if manual_pool:
            all_hits.extend(
                rerank_chunks_with_scores(
                    query,
                    manual_pool,
                    top_n=min(m_quota * 2, len(manual_pool)),
                )
            )
        if log_pool:
            all_hits.extend(
                rerank_chunks_with_scores(
                    query,
                    log_pool,
                    top_n=min(l_quota * 2, len(log_pool)),
                )
            )
        if not all_hits:
            return []

        all_hits.sort(key=lambda h: -h.score)

        max_m, max_l = merge_quota_for_route(
            route, final_topk, cross_max=MERGE_MAX_CROSS_SOURCE
        )

        passed: list = []
        n_m = n_l = 0
        seen: set[str] = set()
        for h in all_hits:
            if h.score < threshold:
                continue
            cid = h.chunk.chunk_id
            if cid in seen:
                continue
            if h.chunk.source == "manual" and n_m >= max_m:
                continue
            if h.chunk.source == "log" and n_l >= max_l:
                continue
            passed.append(h)
            seen.add(cid)
            if h.chunk.source == "manual":
                n_m += 1
            else:
                n_l += 1
            if len(passed) >= final_topk:
                break

        if fill_shortage and len(passed) < final_topk:
            for h in all_hits:
                cid = h.chunk.chunk_id
                if cid in seen:
                    continue
                passed.append(h)
                seen.add(cid)
                if len(passed) >= final_topk:
                    break

        return passed[:final_topk]

    def ask(
        self,
        query: str,
        *,
        context_n: int = FINAL_CONTEXT_N,
        topk_bm25: int = TOPK_BM25,
        topk_vec: int = TOPK_VECTOR,
        merge_pool_size: int = 80,
        rerank_topk: int = 50,
        threshold: float = 0.7,
        final_topk: int = 20,
        fill_shortage: bool = True,
    ) -> DualRagAnswer:
        retrieved = self.retrieve(
            query,
            topk_bm25=topk_bm25,
            topk_vec=topk_vec,
            merge_pool_size=merge_pool_size,
            rerank_topk=rerank_topk,
            threshold=threshold,
            final_topk=final_topk,
            fill_shortage=fill_shortage,
        )
        route = retrieved.query_route
        if not retrieved.recall_hits:
            return DualRagAnswer(
                answer="未召回到足够相关的参考资料，无法生成可靠回答。",
                query_route=route,
                citations=[],
                recall_hits=[],
            )

        take_n = min(context_n, len(retrieved.recall_hits))
        if take_n < MIN_CONTEXT_N and len(retrieved.recall_hits) >= MIN_CONTEXT_N:
            take_n = MIN_CONTEXT_N
        take_n = min(take_n, len(retrieved.recall_hits))
        chunks = [h.chunk for h in retrieved.recall_hits[:take_n]]
        ctx_blocks = expand_context(self.store, chunks)
        user_msg = build_user_message(query, ctx_blocks)
        answer = chat(system_prompt_for_route(route), user_msg)
        cites = [c.chunk_id for c in chunks]
        n_manual = sum(1 for c in chunks if c.source == "manual")
        n_log = sum(1 for c in chunks if c.source == "log")
        return DualRagAnswer(
            answer=answer,
            query_route=route,
            citations=cites,
            recall_hits=retrieved.recall_hits,
            manual_in_context=n_manual,
            log_in_context=n_log,
        )
