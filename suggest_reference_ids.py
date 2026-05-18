"""
为 qa_query.json 自动生成 reference_id 候选，便于快速构建标准问答集。

用法：
  python suggest_reference_ids.py --input qa_query.json --output qa_query_with_candidates.json --topk 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.pipeline import RagPipeline


def _to_candidate(chunk) -> dict:
    return {
        "reference_id": chunk.chunk_id,
        "source_type": chunk.source,
        "doc_id": chunk.meta.get("doc_id", ""),
        "chapter_path": chunk.meta.get("chapter_path", ""),
        "page_number": chunk.meta.get("page_range", ""),
        "case_id": chunk.meta.get("case_id", ""),
        "preview": chunk.text[:180].replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="qa_query.json")
    parser.add_argument("--output", type=str, default="qa_query_with_candidates.json")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    qa_list = json.loads(input_path.read_text(encoding="utf-8"))

    pipe = RagPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    for item in qa_list:
        query = str(item.get("query", "")).strip()
        if not query:
            item["retrieval_candidates"] = []
            continue

        route, chunks = pipe.retrieve(query, pool_size=max(args.topk, 1))
        item["retrieval_route"] = route
        item["retrieval_candidates"] = [_to_candidate(c) for c in chunks[: args.topk]]

    output_path.write_text(
        json.dumps(qa_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成候选 reference_id 文件: {output_path}")


if __name__ == "__main__":
    main()

