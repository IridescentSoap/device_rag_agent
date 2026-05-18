"""仅日志检索链路：
BM25 + 向量召回 -> RRF 融合 -> Reranker -> 阈值过滤。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import INDEX_DIR, RERANK_MODEL, TOPK_BM25, TOPK_VECTOR
from rag.embedder import BgeEmbedder
from rag.index_store import DualIndexStore
from rag.prompts import log_chunk_body_for_prompt
from rag.rerank import RerankHit, rerank_chunks_with_scores
from rag.retrieve import retrieve_from_corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("-q", "--query", type=str, default="川大自动化系统FDP进程宕掉", help="查询文本")
    p.add_argument("--topk-bm25", type=int, default=TOPK_BM25)
    p.add_argument("--topk-vec", type=int, default=TOPK_VECTOR)
    p.add_argument("--pool-size", type=int, default=100, help="RRF 融合后候选池大小")
    p.add_argument("--rerank-topk", type=int, default=20, help="送入阈值过滤前的 rerank TopK")
    p.add_argument("--threshold", type=float, default=0.7, help="rerank 分数阈值")
    p.add_argument("--final-topk", type=int, default=10, help="最终输出条数上限")
    p.add_argument(
        "--fill-shortage",
        action="store_true",
        help="阈值过滤后候选不足时，按 rerank 分数高到低自动补齐到 final-topk",
    )
    p.add_argument("--preview", type=int, default=160, help="文本预览长度")
    args = p.parse_args()

    store = DualIndexStore.load(Path(args.index_dir))
    if not store.log.chunks:
        print("日志语料为空，请先执行 build_index.py 构建日志索引。")
        return

    from rag import config

    model_name = args.embedding_model or config.EMBEDDING_MODEL
    embedder = BgeEmbedder(model_name, device=args.device)

    # 仅对 log 语料做 BM25+向量双路召回，并在单语料内做 RRF 融合。
    merged_scores = retrieve_from_corpus(
        store.log,
        args.query,
        embedder,
        store.client,
        topk_bm25=args.topk_bm25,
        topk_vec=args.topk_vec,
    )
    if not merged_scores:
        print("未召回到候选。")
        return

    id_map = {c.chunk_id: c for c in store.log.chunks}
    rrf_ranked = sorted(merged_scores.items(), key=lambda x: -x[1])[: args.pool_size]
    pool = [id_map[cid] for cid, _ in rrf_ranked if cid in id_map]
    if not pool:
        print("候选池为空。")
        return

    rerank_hits: list[RerankHit] = rerank_chunks_with_scores(
        args.query,
        pool,
        top_n=min(args.rerank_topk, len(pool)),
        model=RERANK_MODEL,
    )
    passed = [h for h in rerank_hits if h.score >= args.threshold][: args.final_topk]
    if args.fill_shortage and len(passed) < args.final_topk:
        selected_ids = {h.chunk.chunk_id for h in passed}
        for h in rerank_hits:
            if h.chunk.chunk_id in selected_ids:
                continue
            passed.append(h)
            selected_ids.add(h.chunk.chunk_id)
            if len(passed) >= args.final_topk:
                break

    print(f"query: {args.query}")
    print(f"log_chunks: {len(store.log.chunks)}")
    print(f"rrf_pool: {len(pool)}")
    print(f"rerank_returned: {len(rerank_hits)}")
    print(f"threshold: {args.threshold}")
    print(f"fill_shortage: {args.fill_shortage}")
    print(f"final_candidates: {len(passed)}")
    print("-" * 90)
    for i, h in enumerate(passed, start=1):
        c = h.chunk
        block = log_chunk_body_for_prompt(c)
        text = block.replace("\n", " ").strip()
        if len(text) > args.preview:
            text = text[: args.preview] + "..."
        case_id = c.meta.get("case_id", "")
        print(f"[{i}] score={h.score:.4f} chunk_id={c.chunk_id} case_id={case_id}")
        print(f"    context: {text}")
        print("-" * 90)


if __name__ == "__main__":
    main()
