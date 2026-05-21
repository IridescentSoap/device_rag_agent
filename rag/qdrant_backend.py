"""Qdrant 向量存储：与 BM25 并行，稠密向量仅存于 Qdrant。"""

from __future__ import annotations

import uuid
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from rag.schemas import ChunkRecord

_POINT_NAMESPACE = uuid.UUID("8c5e3f2a-9b1d-5e4c-a7f6-2d8e1b3c4a5f")
_QDRANT_WARNED = False


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def get_client(url: str, api_key: str | None = None) -> QdrantClient:
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


def _collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        cols = client.get_collections().collections
        return any(c.name == name for c in cols)
    except Exception:
        return False


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    if recreate and _collection_exists(client, name):
        client.delete_collection(collection_name=name)
    if not _collection_exists(client, name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[ChunkRecord],
    vectors: np.ndarray,
    batch_size: int = 64,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks 与 vectors 长度不一致")
    points: list[PointStruct] = []
    for c, row in zip(chunks, vectors, strict=True):
        pid = stable_point_id(c.chunk_id)
        vec = row.tolist() if hasattr(row, "tolist") else list(row)
        points.append(
            PointStruct(
                id=pid,
                vector=vec,
                payload={
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "doc_id": str((c.meta or {}).get("doc_id") or ""),
                    "record": c.to_dict(),
                },
            )
        )
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)


def search_by_vector(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    limit: int,
    source_filter: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[tuple[ChunkRecord, float]]:
    global _QDRANT_WARNED

    qv = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    must: list[FieldCondition] = []
    should: list[FieldCondition] = []
    if source_filter:
        must.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if doc_ids:
        doc_conds = [
            FieldCondition(key="doc_id", match=MatchValue(value=did)) for did in doc_ids
        ]
        if len(doc_ids) == 1:
            must.extend(doc_conds)
        else:
            should.extend(doc_conds)
    flt = None
    if must or should:
        flt = Filter(must=must or None, should=should or None)
    try:
        hits = None
        # 兼容不同 qdrant-client 版本：
        # - 旧版常用 client.search(...)
        # - 新版可能改为 client.query_points(...)
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=collection,
                query_vector=qv,
                limit=limit,
                query_filter=flt,
                with_payload=True,
            )
        elif hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=collection,
                query=qv,
                limit=limit,
                query_filter=flt,
                with_payload=True,
            )
            if isinstance(res, list):
                hits = res
            else:
                hits = getattr(res, "points", None) or getattr(res, "result", None) or []
        else:
            return []
    except Exception as e:
        if not _QDRANT_WARNED:
            print(f"[WARN] Qdrant 检索失败，已回退为 BM25-only: {e}")
            _QDRANT_WARNED = True
        return []

    out: list[tuple[ChunkRecord, float]] = []
    for h in hits:
        pl = h.payload or {}
        rec = pl.get("record")
        if not rec:
            continue
        score_raw = getattr(h, "score", None)
        if score_raw is None:
            score_raw = getattr(h, "distance", None)
        score = float(score_raw) if score_raw is not None else 0.0
        out.append((ChunkRecord.from_dict(rec), score))
    return out


def delete_collection_if_exists(client: QdrantClient, name: str) -> None:
    if _collection_exists(client, name):
        client.delete_collection(collection_name=name)
