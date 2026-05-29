"""检索 hit 合并工具。"""

from __future__ import annotations

from rag.rerank import RerankHit


def merge_hits(*hit_lists: list[RerankHit]) -> list[RerankHit]:
    """按 chunk_id 去重，保留更高 score，最终按 score 降序。"""
    by_id: dict[str, RerankHit] = {}
    for batch in hit_lists:
        for h in batch:
            cid = h.chunk.chunk_id
            prev = by_id.get(cid)
            if prev is None or float(h.score) > float(prev.score):
                by_id[cid] = h
    return sorted(by_id.values(), key=lambda x: float(x.score), reverse=True)
