"""
快速验证「双路召回」（BM25 + 向量 → RRF）相对单路的效果。

模式 1：log_self（默认）
  从日志索引中随机抽样若干条，用截断后的文本当作查询，正样本为该条 chunk_id。
  用于冒烟与粗看 Recall@K（自检索偏乐观，真实业务请用模式 2）。

模式 2：--eval-file eval.jsonl
  每行 JSON：{"query": "...", "gold_chunk_ids": ["log_case_1"]} 或单个 "gold_chunk_id"。

用法：
  set QDRANT_URL=http://localhost:6333
  python build_index.py --logs data/filtered_maintenance_data.csv
  python eval_retrieval_quick.py --sample 200 --seed 42
  python eval_retrieval_quick.py --eval-file data/eval_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rag.config import INDEX_DIR, TOPK_PER_SOURCE, TOPK_VECTOR
from rag.embedder import BgeEmbedder
from rag.index_store import DualIndexStore
from rag.retrieval_modes import (
    mrr,
    ranked_bm25_only,
    ranked_rrf_single_corpus,
    ranked_vector_only,
    recall_at_k,
)


def _load_tasks_log_self(store: DualIndexStore, sample: int, seed: int, query_chars: int) -> list[tuple[str, set[str]]]:
    chunks = store.log.chunks
    if not chunks:
        return []
    rng = random.Random(seed)
    idxs = list(range(len(chunks)))
    rng.shuffle(idxs)
    idxs = idxs[: min(sample, len(idxs))]
    tasks: list[tuple[str, set[str]]] = []
    for i in idxs:
        c = chunks[i]
        gold = {c.chunk_id}
        text = c.text.strip()
        q = text[:query_chars] if len(text) > query_chars else text
        if len(q) < 8:
            q = text
        tasks.append((q, gold))
    return tasks


def _load_tasks_file(path: Path) -> list[tuple[str, set[str]]]:
    tasks: list[tuple[str, set[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["query"]
            g: set[str] = set()
            if "gold_chunk_ids" in obj:
                g.update(obj["gold_chunk_ids"])
            if "gold_chunk_id" in obj:
                g.add(obj["gold_chunk_id"])
            if not g:
                continue
            tasks.append((q, g))
    return tasks


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--mode", choices=("log_self", "file"), default="log_self")
    p.add_argument("--eval-file", type=str, default=None)
    p.add_argument("--sample", type=int, default=100, help="log_self 抽样条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-chars", type=int, default=120, help="log_self 查询取每条文本前 N 字")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10], help="Recall@k 的 k 列表")
    p.add_argument("--topk-bm25", type=int, default=None)
    p.add_argument("--topk-vec", type=int, default=None)
    p.add_argument("--final-k", type=int, default=20, help="RRF 输出列表截断长度（用于 MRR）")
    args = p.parse_args()

    from rag import config

    topk_bm = args.topk_bm25 or TOPK_PER_SOURCE
    topk_v = args.topk_vec or TOPK_VECTOR
    final_k = max(args.final_k, max(args.ks))

    store = DualIndexStore(Path(args.index_dir))
    model = args.embedding_model or config.EMBEDDING_MODEL
    embedder = BgeEmbedder(model, device=args.device)

    if args.mode == "file":
        if not args.eval_file:
            raise SystemExit("file 模式需要 --eval-file")
        tasks = _load_tasks_file(Path(args.eval_file))
    else:
        tasks = _load_tasks_log_self(
            store, args.sample, args.seed, args.query_chars
        )

    if not tasks:
        raise SystemExit("无评测任务：请确认已建索引且日志索引非空，或检查 eval 文件。")

    corpus = store.log
    ks = args.ks

    rows_bm: dict[int, list[float]] = {k: [] for k in ks}
    rows_vec: dict[int, list[float]] = {k: [] for k in ks}
    rows_rrf: dict[int, list[float]] = {k: [] for k in ks}
    mrr_bm: list[float] = []
    mrr_vec: list[float] = []
    mrr_rrf: list[float] = []

    for q, gold in tasks:
        lb = ranked_bm25_only(corpus, q, max(final_k, max(ks)))
        lv = ranked_vector_only(corpus, q, embedder, store.client, max(final_k, max(ks)))
        lr = ranked_rrf_single_corpus(
            corpus, q, embedder, store.client, topk_bm, topk_v, final_k
        )
        for k in ks:
            rows_bm[k].append(recall_at_k(gold, lb, k))
            rows_vec[k].append(recall_at_k(gold, lv, k))
            rows_rrf[k].append(recall_at_k(gold, lr, k))
        mrr_bm.append(mrr(gold, lb))
        mrr_vec.append(mrr(gold, lv))
        mrr_rrf.append(mrr(gold, lr))

    n = len(tasks)
    print(f"评测条数: {n}（corpus=log，双路=BM25+向量→RRF）")
    print(f"RRF 参数: topk_bm25={topk_bm}, topk_vec={topk_v}, 取前 {final_k} 算 MRR")
    print()
    for name, rdict, m in (
        ("BM25 单路", rows_bm, mrr_bm),
        ("向量 单路", rows_vec, mrr_vec),
        ("双路 RRF", rows_rrf, mrr_rrf),
    ):
        line = [f"Recall@{k}={_mean(rdict[k]):.4f}" for k in ks]
        print(f"{name}: " + ", ".join(line) + f", MRR={_mean(m):.4f}")

    print()
    print("说明：log_self 为自检索近似，Recall 往往偏高；上线前请用 --eval-file 人工标注集复核。")


if __name__ == "__main__":
    main()
