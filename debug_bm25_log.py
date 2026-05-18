# debug_bm25_log.py
# 用法：
#   python debug_bm25_log.py -q "你的查询" --topk 10 --index-dir /root/autodl-tmp/device_rag_llm/data/rag_index

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rag.index_store import DualIndexStore


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=int)
    n = min(k, scores.size)
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", type=str, required=True, help="查询文本")
    p.add_argument("--topk", type=int, default=10, help="输出前K条")
    p.add_argument(
        "--index-dir",
        type=str,
        default="/root/autodl-tmp/device_rag_llm/data/rag_index",
        help="build_index 保存的索引目录",
    )
    p.add_argument("--preview", type=int, default=120, help="文本预览长度")
    args = p.parse_args()

    store = DualIndexStore.load(Path(args.index_dir))
    corpus = store.log  # 只看 log 语料

    if not corpus.chunks:
        print("log 语料为空，请先执行 build_index.py")
        return

    scores = corpus.bm25_scores(args.query)
    idxs = topk_indices(scores, args.topk)

    print(f"query: {args.query}")
    print(f"log chunks: {len(corpus.chunks)}")
    print(f"topk: {len(idxs)}")
    print("-" * 80)

    for rank, i in enumerate(idxs, start=1):
        c = corpus.chunks[int(i)]
        s = float(scores[int(i)])
        text = c.text.replace("\n", " ").strip()
        if len(text) > args.preview:
            text = text[: args.preview] + "..."
        print(f"[{rank}] score={s:.6f} chunk_id={c.chunk_id} source={c.source}")
        print(f"    text: {text}")
        print("-" * 80)


if __name__ == "__main__":
    main()