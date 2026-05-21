"""双源索引：BM25（本地）+ 稠密向量（Qdrant）；chunks 元数据落盘 jsonl。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embedder import BgeEmbedder
from rag.manual_catalog import ManualDocCatalog, build_manual_catalog
from rag.qdrant_backend import ensure_collection, get_client, search_by_vector, upsert_chunks
from rag.schemas import ChunkRecord

from qdrant_client import QdrantClient


def _tokenize_zh(text: str) -> list[str]:
    import jieba

    return list(jieba.cut(text.strip().lower()))


class CorpusIndex:
    def __init__(self, source: str):
        self.source = source
        self.chunks: list[ChunkRecord] = []
        self.parents: dict[str, ChunkRecord] = {}
        self.doc_catalog: ManualDocCatalog | None = None
        self._bm25: BM25Okapi | None = None
        self.qdrant_collection: str = ""

    def _build_bm25(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize_zh(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def set_parents(self, parents: list[ChunkRecord] | dict[str, ChunkRecord]) -> None:
        if isinstance(parents, dict):
            self.parents = parents
        else:
            self.parents = {p.chunk_id: p for p in parents}

    def build(
        self,
        chunks: list[ChunkRecord],
        embedder: BgeEmbedder,
        client: QdrantClient,
        collection_name: str,
        recreate: bool = False,
        *,
        parents: list[ChunkRecord] | dict[str, ChunkRecord] | None = None,
    ) -> None:
        if parents is not None:
            self.set_parents(parents)
        self.chunks = chunks
        if self.source == "manual" and chunks:
            parent_list = (
                list(parents.values()) if isinstance(parents, dict) else (parents or [])
            )
            self.doc_catalog = build_manual_catalog(chunks, parent_list)
        self.qdrant_collection = collection_name
        if not chunks:
            self._bm25 = None
            return
        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts)
        dim = int(vectors.shape[1])
        ensure_collection(client, collection_name, vector_size=dim, recreate=recreate)
        upsert_chunks(client, collection_name, chunks, vectors)
        self._build_bm25()

    def bm25_scores(self, query: str) -> np.ndarray:
        if not self._bm25 or not self.chunks:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._bm25.get_scores(_tokenize_zh(query)), dtype=np.float64)

    def chunks_for_doc_ids(self, doc_ids: list[str] | None) -> list[ChunkRecord]:
        if not doc_ids:
            return self.chunks
        allow = set(doc_ids)
        return [c for c in self.chunks if c.meta.get("doc_id") in allow]

    def vector_search(
        self,
        query: str,
        embedder: BgeEmbedder,
        client: QdrantClient,
        topk: int,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkRecord]:
        pool = self.chunks_for_doc_ids(doc_ids)
        if not pool or not self.qdrant_collection:
            return []
        qv = embedder.encode([query])[0]
        hits = search_by_vector(
            client,
            self.qdrant_collection,
            qv,
            limit=min(topk, len(pool)),
            source_filter=self.source,
            doc_ids=doc_ids,
        )
        return [c for c, _ in hits]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        meta_path = dir_path / "chunks.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        parents_path = dir_path / "parents.jsonl"
        if self.parents:
            with parents_path.open("w", encoding="utf-8") as f:
                for p in self.parents.values():
                    f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        elif parents_path.exists():
            parents_path.unlink()
        docs_path = dir_path / "docs.jsonl"
        if self.source == "manual" and self.doc_catalog and self.doc_catalog.docs:
            self.doc_catalog.save(docs_path)
        elif docs_path.exists():
            docs_path.unlink()
        meta = {
            "qdrant_collection": self.qdrant_collection,
            "source": self.source,
            "has_parents": bool(self.parents),
            "manual_doc_count": len(self.doc_catalog.docs) if self.doc_catalog else 0,
        }
        (dir_path / "index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path, source: str) -> "CorpusIndex":
        idx = cls(source=source)
        meta_file = dir_path / "index_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            idx.qdrant_collection = str(meta.get("qdrant_collection", ""))
        meta_path = dir_path / "chunks.jsonl"
        if not meta_path.exists():
            return idx
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx.chunks.append(ChunkRecord.from_dict(json.loads(line)))
        parents_path = dir_path / "parents.jsonl"
        if parents_path.exists():
            with parents_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = ChunkRecord.from_dict(json.loads(line))
                    idx.parents[p.chunk_id] = p
        if source == "manual":
            docs_path = dir_path / "docs.jsonl"
            if docs_path.exists():
                idx.doc_catalog = ManualDocCatalog.load(docs_path)
            elif idx.chunks:
                idx.doc_catalog = build_manual_catalog(
                    idx.chunks, list(idx.parents.values())
                )
        idx._build_bm25()
        return idx


class DualIndexStore:
    def __init__(self, root: Path | str, client: QdrantClient | None = None):
        from rag import config

        self.root = Path(root)
        url = config.QDRANT_URL
        key = config.QDRANT_API_KEY or None
        self.client = client or get_client(url, key)
        self.manual = CorpusIndex("manual")
        self.log = CorpusIndex("log")

    def paths(self) -> tuple[Path, Path]:
        return self.root / "manual", self.root / "log"

    def collection_names(self, prefix: str) -> tuple[str, str]:
        safe = prefix.strip().replace(" ", "_")[:80]
        return f"{safe}_manual", f"{safe}_log"

    def save(self) -> None:
        m, l = self.paths()
        self.manual.save(m)
        self.log.save(l)

    @classmethod
    def load(cls, root: Path | str, client: QdrantClient | None = None) -> "DualIndexStore":
        store = cls(root, client=client)
        m, l = store.paths()
        store.manual = CorpusIndex.load(m, "manual")
        store.log = CorpusIndex.load(l, "log")
        return store
