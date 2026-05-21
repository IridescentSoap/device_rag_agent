"""双源问答入口（兼容旧名）：等同 query_dual_rag.py 全流程。"""

from __future__ import annotations

import argparse
import os

from rag.config import FINAL_CONTEXT_N, INDEX_DIR, TOPK_BM25, TOPK_VECTOR
from rag.pipeline import DualSourceRagPipeline


def main() -> None:
    p = argparse.ArgumentParser(description="双源 RAG（路由 + 召回 + 生成）")
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("-q", "--query", type=str, default="", help="单次提问；省略则进入循环")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--context-n", type=int, default=FINAL_CONTEXT_N)
    p.add_argument("--show-recall", action="store_true")
    args = p.parse_args()

    if not (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    ):
        print("警告: 未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，生成阶段会失败。")

    pipe = DualSourceRagPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    def run_one(q: str) -> None:
        out = pipe.ask(q, threshold=args.threshold, context_n=args.context_n)
        if args.show_recall:
            print(f"召回 {len(out.recall_hits)} 条，路由={out.query_route}")
        print("\n--- 路由 ---\n", out.query_route)
        print("\n--- 引用 chunk_id ---\n", out.citations)
        print("\n--- 回答 ---\n", out.answer)

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
