"""双源全流程 RAG：manual + log 召回 -> query 路由 -> 融合 rerank -> LLM 生成。"""

from __future__ import annotations

import argparse
import os

from rag.config import FINAL_CONTEXT_N, INDEX_DIR, TOPK_BM25, TOPK_VECTOR
from rag.pipeline import DualSourceRagPipeline
from rag.prompts import log_chunk_body_for_prompt


def print_recall(pipe: DualSourceRagPipeline, hits, preview: int) -> None:
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        if c.source == "log":
            text = log_chunk_body_for_prompt(c).replace("\n", " ").strip()
        else:
            text = c.text.replace("\n", " ").strip()
        if len(text) > preview:
            text = text[:preview] + "..."
        extra = ""
        if c.meta.get("case_id"):
            extra += f" case_id={c.meta.get('case_id')}"
        if c.meta.get("page_range"):
            extra += f" page={c.meta.get('page_range')}"
        if c.meta.get("doc_id"):
            extra += f" doc_id={c.meta.get('doc_id')}"
        print(f"[{i}] score={h.score:.4f} source={c.source} chunk_id={c.chunk_id}{extra}")
        print(f"    context: {text}")
        print("-" * 90)


def main() -> None:
    p = argparse.ArgumentParser(
        description="双源 RAG：手册 + 日志召回、query 路由、生成完整回答",
    )
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("-q", "--query", type=str, default="", help="单次提问；省略则进入交互循环")
    p.add_argument("--topk-bm25", type=int, default=TOPK_BM25)
    p.add_argument("--topk-vec", type=int, default=TOPK_VECTOR)
    p.add_argument("--merge-pool-size", type=int, default=80, help="路由加权融合后的候选池大小")
    p.add_argument("--rerank-topk", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.7, help="rerank 分数阈值")
    p.add_argument("--final-topk", type=int, default=20, help="阈值过滤后保留条数上限")
    p.add_argument("--context-n", type=int, default=FINAL_CONTEXT_N, help="送入 LLM 的 chunk 数")
    p.add_argument(
        "--fill-shortage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="阈值过滤不足时按 rerank 分补齐",
    )
    p.add_argument("--retrieve-only", action="store_true", help="仅检索与路由，不生成")
    p.add_argument("--show-recall", action="store_true", help="生成时同时打印召回明细")
    p.add_argument("--preview", type=int, default=200)
    args = p.parse_args()

    if not args.retrieve_only and not (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    ):
        print("警告: 未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，生成阶段会失败。")

    pipe = DualSourceRagPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )
    if not pipe.store.manual.chunks and not pipe.store.log.chunks:
        print("手册与日志语料均为空，请先执行 build_index.py。")
        return

    ask_kw = dict(
        topk_bm25=args.topk_bm25,
        topk_vec=args.topk_vec,
        merge_pool_size=args.merge_pool_size,
        rerank_topk=args.rerank_topk,
        threshold=args.threshold,
        final_topk=args.final_topk,
        fill_shortage=args.fill_shortage,
        context_n=args.context_n,
    )
    retrieve_kw = {k: v for k, v in ask_kw.items() if k != "context_n"}

    def run_one(q: str) -> None:
        if args.retrieve_only:
            ret = pipe.retrieve(q, **retrieve_kw)
            print(f"query: {q}")
            print(f"manual_rrf_hits: {ret.manual_rrf_pool}")
            print(f"log_rrf_hits: {ret.log_rrf_pool}")
            print(f"merged_pool: {ret.merged_pool}")
            print(f"query_route: {ret.query_route}")
            print(f"final_candidates: {len(ret.recall_hits)}")
            if ret.recall_hits:
                print("-" * 90)
                print_recall(pipe, ret.recall_hits, args.preview)
            return

        out = pipe.ask(q, **ask_kw)
        if args.show_recall and out.recall_hits:
            print(f"query: {q}")
            print(f"query_route: {out.query_route}")
            print(
                f"recall: {len(out.recall_hits)} | "
                f"context: manual={out.manual_in_context} log={out.log_in_context}"
            )
            print("-" * 90)
            print_recall(pipe, out.recall_hits, args.preview)

        print("\n--- 路由 ---\n", out.query_route)
        print(
            "\n--- 上下文来源 ---\n",
            f"manual={out.manual_in_context}, log={out.log_in_context}",
        )
        print("\n--- 引用 chunk_id ---\n", out.citations)
        print("\n--- 回答 ---\n", out.answer)

    if args.query:
        run_one(args.query)
        return

    print("双源 RAG（空行退出）:")
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
