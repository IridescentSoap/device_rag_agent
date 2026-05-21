"""
重建索引后，按旧 chunk 文本相似度将评测集 gold_chunk_ids 映射到新 chunk_id。

用法：
  cp data/eval/business_eval_30.jsonl data/eval/business_eval_30.jsonl.bak
  python scripts/remap_eval_gold.py \\
    --old-chunks data/rag_index/manual/chunks.jsonl.bak \\
    --new-chunks data/rag_index/manual/chunks.jsonl \\
    --eval data/eval/business_eval_30.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag.schemas import ChunkRecord


def _tokens(text: str) -> set[str]:
    import jieba

    return {t for t in jieba.cut(text.lower()) if len(t.strip()) > 1}


def load_chunks(path: Path) -> dict[str, ChunkRecord]:
    out: dict[str, ChunkRecord] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = ChunkRecord.from_dict(json.loads(line))
            out[c.chunk_id] = c
    return out


def best_match(old_text: str, candidates: dict[str, ChunkRecord]) -> str | None:
    old_t = _tokens(old_text)
    if not old_t:
        return None
    best_id = None
    best_score = 0.0
    for cid, c in candidates.items():
        new_t = _tokens(c.text)
        if not new_t:
            continue
        inter = len(old_t & new_t)
        union = len(old_t | new_t)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_id = cid
    if best_score < 0.25:
        return None
    return best_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-chunks", type=Path, required=True)
    p.add_argument("--new-chunks", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    old_map = load_chunks(args.old_chunks)
    new_map = load_chunks(args.new_chunks)
    out_path = args.out or args.eval

    lines_out: list[str] = []
    with args.eval.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            new_golds: list[str] = []
            for gid in obj.get("gold_chunk_ids", []):
                old = old_map.get(gid)
                if old is not None:
                    matched = best_match(old.text, new_map)
                    if matched:
                        new_golds.append(matched)
                    elif gid in new_map:
                        new_golds.append(gid)
                    continue
                if gid in new_map:
                    new_golds.append(gid)
                    continue
                new_golds.append(gid)
            # 去重保序
            seen: set[str] = set()
            deduped: list[str] = []
            for g in new_golds:
                if g not in seen:
                    seen.add(g)
                    deduped.append(g)
            obj["gold_chunk_ids"] = deduped
            lines_out.append(json.dumps(obj, ensure_ascii=False))

    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
