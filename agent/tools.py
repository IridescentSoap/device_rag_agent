"""封装现有 RAG 检索与生成能力为 Agent 工具。"""



from __future__ import annotations



from contextlib import nullcontext

from typing import TYPE_CHECKING



from agent.fast_mode import (

    fast_mode_summary,

    fast_rerank_env,

    get_fast_settings,

    resolve_fast_mode,

)

from rag.config import (

    EMBEDDING_DEVICE,

    FINAL_CONTEXT_N,

    INDEX_DIR,

    PER_SOURCE_POOL,

    TOPK_BM25,

    TOPK_VECTOR,

)

from rag.context_expand import expand_context

from rag.llm import chat

from rag.prompts import (

    SYSTEM_LOG_RAG,

    SYSTEM_MANUAL_RAG,

    SYSTEM_RAG,

    build_user_message,

    system_prompt_for_route,

)

from rag.pipeline import DualSourceRagPipeline

from rag.rerank import RerankHit, rerank_chunks_with_scores

from rag.retrieve import (

    chunks_from_corpus_scores,

    dedupe_log_chunks,

    resolve_manual_doc_ids,

    retrieve_from_corpus,

)



if TYPE_CHECKING:

    from rag.schemas import ChunkRecord





def _hits_to_chunks(hits: list[RerankHit]) -> list[ChunkRecord]:

    return [h.chunk for h in hits]





class RagTools:

    """懒加载双源 Pipeline，对外提供 search / generate 工具。"""



    def __init__(

        self,

        index_dir: str | None = None,

        device: str | None = None,

        *,

        fast_mode: bool | None = None,

    ):

        self.index_dir = index_dir or str(INDEX_DIR)

        self.device = device or EMBEDDING_DEVICE

        self.fast_mode = resolve_fast_mode(fast_mode)

        self._fast = get_fast_settings() if self.fast_mode else None

        self._pipe: DualSourceRagPipeline | None = None



    def _rerank_ctx(self):

        return fast_rerank_env() if self.fast_mode else nullcontext()



    @property

    def pipe(self) -> DualSourceRagPipeline:

        if self._pipe is None:

            self._pipe = DualSourceRagPipeline(

                index_dir=self.index_dir,

                device=self.device,

            )

        return self._pipe



    def search_manual(

        self,

        query: str,

        *,

        pool_size: int | None = None,

        top_k: int | None = None,

        threshold: float = 0.7,

    ) -> list[RerankHit]:

        if self._fast:

            pool_size = pool_size if pool_size is not None else self._fast.manual_pool_size

            top_k = top_k if top_k is not None else self._fast.manual_top_k

        else:

            pool_size = pool_size if pool_size is not None else PER_SOURCE_POOL

            top_k = top_k if top_k is not None else 12



        store = self.pipe.store

        doc_ids = resolve_manual_doc_ids(store.manual, query)

        scores = retrieve_from_corpus(

            store.manual,

            query,

            self.pipe.embedder,

            store.client,

            topk_bm25=TOPK_BM25,

            topk_vec=TOPK_VECTOR,

            doc_ids=doc_ids,

        )

        pool = chunks_from_corpus_scores(store.manual, scores, pool_size)

        if not pool:

            return []

        with self._rerank_ctx():

            hits = rerank_chunks_with_scores(

                query, pool, top_n=min(top_k * 2, len(pool))

            )

        return [h for h in hits if h.score >= threshold][:top_k]



    def search_logs(

        self,

        query: str,

        *,

        pool_size: int | None = None,

        top_k: int | None = None,

        threshold: float = 0.7,

    ) -> list[RerankHit]:

        if self._fast:

            pool_size = pool_size if pool_size is not None else self._fast.log_pool_size

            top_k = top_k if top_k is not None else self._fast.log_top_k

        else:

            pool_size = pool_size if pool_size is not None else PER_SOURCE_POOL

            top_k = top_k if top_k is not None else 12



        store = self.pipe.store

        scores = retrieve_from_corpus(

            store.log,

            query,

            self.pipe.embedder,

            store.client,

            topk_bm25=TOPK_BM25,

            topk_vec=TOPK_VECTOR,

        )

        pool = dedupe_log_chunks(

            chunks_from_corpus_scores(store.log, scores, pool_size)

        )

        if not pool:

            return []

        with self._rerank_ctx():

            hits = rerank_chunks_with_scores(

                query, pool, top_n=min(top_k * 2, len(pool))

            )

        return [h for h in hits if h.score >= threshold][:top_k]



    def hybrid_search(

        self,

        query: str,

        *,

        final_topk: int | None = None,

        threshold: float = 0.7,

    ) -> tuple[list[RerankHit], str]:

        if self._fast:

            f = self._fast

            final_topk = final_topk if final_topk is not None else f.final_topk

            retrieve_kw = dict(

                per_source_pool=f.per_source_pool,

                rerank_topk=f.rerank_topk,

                final_topk=final_topk,

            )

        else:

            final_topk = final_topk if final_topk is not None else 20

            retrieve_kw = dict(final_topk=final_topk)



        with self._rerank_ctx():

            ret = self.pipe.retrieve(

                query,

                topk_bm25=TOPK_BM25,

                topk_vec=TOPK_VECTOR,

                threshold=threshold,

                fill_shortage=True,

                **retrieve_kw,

            )

        return list(ret.recall_hits), ret.query_route



    def generate_answer(

        self,

        query: str,

        hits: list[RerankHit],

        *,

        agent_route: str = "hybrid_diagnosis",

        pipeline_route: str | None = None,

        context_n: int | None = None,

        sub_queries: list[str] | None = None,

    ) -> tuple[str, list[str]]:

        if not hits:

            return (

                "未找到足够相关的参考资料，无法给出可靠结论。请补充设备名称、故障现象或手册章节。",

                [],

            )

        if context_n is None:

            context_n = self._fast.context_n if self._fast else FINAL_CONTEXT_N

        chunks = _hits_to_chunks(hits[:context_n])

        ctx_blocks = expand_context(self.pipe.store, chunks)

        pr = pipeline_route or _agent_route_to_pipeline(agent_route)

        system = system_prompt_for_route(pr)

        if agent_route == "manual_query":

            system = SYSTEM_MANUAL_RAG

        elif agent_route == "log_case_query":

            system = SYSTEM_LOG_RAG

        elif agent_route in ("hybrid_diagnosis", "follow_up_query"):

            system = SYSTEM_RAG

        user_msg = build_user_message(query, ctx_blocks, sub_queries=sub_queries)

        answer = chat(system, user_msg)

        cites = [c.chunk_id for c in chunks]

        return answer, cites



    def mode_label(self) -> str:

        return "fast" if self.fast_mode else "standard"



    def mode_config(self) -> dict:

        if self.fast_mode:

            return fast_mode_summary()

        return {"enabled": False}





def _agent_route_to_pipeline(agent_route: str) -> str:

    if agent_route == "manual_query":

        return "manual_heavy"

    if agent_route == "log_case_query":

        return "log_heavy"

    return "balanced"


