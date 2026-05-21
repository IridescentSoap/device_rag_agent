"""仅手册 RAG：BM25 + 向量 -> RRF -> Rerank -> 阈值过滤 -> LLM 生成回答。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag.config import (
    FINAL_CONTEXT_N,
    INDEX_DIR,
    MIN_CONTEXT_N,
    TOPK_BM25,
    TOPK_VECTOR,
)
from rag.embedder import BgeEmbedder
from rag.index_store import DualIndexStore
from rag.llm import chat
from rag.context_expand import expand_context
from rag.prompts import SYSTEM_MANUAL_RAG, build_user_message
from rag.rerank import RerankHit
from rag.retrieval_stage import corpus_retrieve_rerank
from rag.schemas import ChunkRecord


def manual_retrieve(
    store: DualIndexStore,
    embedder: BgeEmbedder,
    query: str,
    *,
    topk_bm25: int,
    topk_vec: int,
    pool_size: int,
    rerank_topk: int,
    threshold: float,
    final_topk: int,
    fill_shortage: bool,
) -> list[RerankHit]:
    return corpus_retrieve_rerank(
        store.manual,
        query,
        embedder,
        store.client,
        topk_bm25=topk_bm25,
        topk_vec=topk_vec,
        pool_size=pool_size,
        rerank_topk=rerank_topk,
        threshold=threshold,
        final_topk=final_topk,
        fill_shortage=fill_shortage,
    )


def manual_ask(
    store: DualIndexStore,
    embedder: BgeEmbedder,
    query: str,
    *,
    topk_bm25: int,
    topk_vec: int,
    pool_size: int,
    rerank_topk: int,
    threshold: float,
    final_topk: int,
    fill_shortage: bool,
    context_n: int,
) -> tuple[str, list[str], list[RerankHit]]:
    passed = manual_retrieve(
        store,
        embedder,
        query,
        topk_bm25=topk_bm25,
        topk_vec=topk_vec,
        pool_size=pool_size,
        rerank_topk=rerank_topk,
        threshold=threshold,
        final_topk=final_topk,
        fill_shortage=fill_shortage,
    )
    if not passed:
        return "未召回到足够相关的参考资料，无法生成可靠回答。", [], []

    take_n = min(context_n, len(passed))
    if take_n < MIN_CONTEXT_N and len(passed) >= MIN_CONTEXT_N:
        take_n = MIN_CONTEXT_N
    take_n = min(take_n, len(passed))
    chunks: list[ChunkRecord] = [h.chunk for h in passed[:take_n]]
    ctx_blocks = expand_context(store, chunks)
    user_msg = build_user_message(query, ctx_blocks)
    answer = chat(SYSTEM_MANUAL_RAG, user_msg)
    cites = [c.chunk_id for c in chunks]
    return answer, cites, passed


def print_recall(hits: list[RerankHit], preview: int) -> None:
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        text = c.text.replace("\n", " ").strip()
        if len(text) > preview:
            text = text[:preview] + "..."
        doc_id = c.meta.get("doc_id", "")
        fname = c.meta.get("file", "")
        page_range = c.meta.get("page_range", "")
        extra = f" doc_id={doc_id}" if doc_id else ""
        if fname:
            extra += f" file={fname}"
        if page_range:
            extra += f" page={page_range}"
        print(f"[{i}] score={h.score:.4f} chunk_id={c.chunk_id}{extra}")
        print(f"    context: {text}")
        print("-" * 90)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "-q",
        "--query",
        type=str,
        default="",
        help="单次提问；省略则进入交互循环",
    )
    p.add_argument("--topk-bm25", type=int, default=TOPK_BM25)
    p.add_argument("--topk-vec", type=int, default=TOPK_VECTOR)
    p.add_argument("--pool-size", type=int, default=100, help="RRF 融合后候选池大小")
    p.add_argument("--rerank-topk", type=int, default=50, help="送入阈值过滤前的 rerank TopK")
    p.add_argument("--threshold", type=float, default=0.7, help="rerank 分数阈值")
    p.add_argument("--final-topk", type=int, default=20, help="阈值过滤后保留条数上限")
    p.add_argument(
        "--context-n",
        type=int,
        default=FINAL_CONTEXT_N,
        help="送入生成模型的 chunk 数（从 final-topk 中取前 N 条）",
    )
    p.add_argument(
        "--fill-shortage",
        action="store_true",
        help="阈值过滤后候选不足时，按 rerank 分数高到低自动补齐到 final-topk",
    )
    p.add_argument(
        "--retrieve-only",
        action="store_true",
        help="仅打印召回结果，不调用生成模型",
    )
    p.add_argument("--show-recall", action="store_true", help="生成回答时同时打印召回明细")
    p.add_argument("--preview", type=int, default=200, help="召回文本预览长度")
    args = p.parse_args()

    if not args.retrieve_only and not (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    ):
        print("警告: 未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，生成阶段会失败。")

    store = DualIndexStore.load(Path(args.index_dir))
    if not store.manual.chunks:
        print("手册语料为空，请先执行 build_index.py 构建手册索引。")
        return

    from rag import config

    model_name = args.embedding_model or config.EMBEDDING_MODEL
    embedder = BgeEmbedder(model_name, device=args.device)

    retrieve_kw = dict(
        topk_bm25=args.topk_bm25,
        topk_vec=args.topk_vec,
        pool_size=args.pool_size,
        rerank_topk=args.rerank_topk,
        threshold=args.threshold,
        final_topk=args.final_topk,
        fill_shortage=args.fill_shortage,
    )

    def run_one(q: str) -> None:
        if args.retrieve_only:
            passed = manual_retrieve(store, embedder, q, **retrieve_kw)
            if not passed:
                print("未召回到候选。")
                return
            print(f"query: {q}")
            print(f"manual_chunks: {len(store.manual.chunks)}")
            print(f"final_candidates: {len(passed)}")
            print("-" * 90)
            print_recall(passed, args.preview)
            return

        answer, cites, passed = manual_ask(
            store,
            embedder,
            q,
            context_n=args.context_n,
            **retrieve_kw,
        )
        if args.show_recall and passed:
            print(f"query: {q}")
            print(f"manual_chunks: {len(store.manual.chunks)}")
            print(f"final_candidates: {len(passed)}")
            print(f"context_n: {min(args.context_n, len(passed))}")
            print("-" * 90)
            print_recall(passed, args.preview)

        print("\n--- 引用 chunk_id ---\n", cites)
        print("\n--- 回答 ---\n", answer)

    if args.query:
        run_one(args.query)
        return

    print("输入问题（空行退出）:")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        run_one(line)


if __name__ == "__main__":
    main()
