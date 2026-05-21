"""多手册目录：建库时生成文档画像，检索前做 doc 级路由。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.schemas import ChunkRecord

_TITLE_RE = re.compile(r"^#{1,2}\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ManualDocProfile:
    doc_id: str
    title: str = ""
    file_name: str = ""
    chunk_count: int = 0
    section_titles: list[str] = field(default_factory=list)
    routing_text: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "file_name": self.file_name,
            "chunk_count": self.chunk_count,
            "section_titles": self.section_titles,
            "routing_text": self.routing_text,
        }

    @staticmethod
    def from_dict(d: dict) -> "ManualDocProfile":
        return ManualDocProfile(
            doc_id=str(d["doc_id"]),
            title=str(d.get("title", "")),
            file_name=str(d.get("file_name", "")),
            chunk_count=int(d.get("chunk_count", 0)),
            section_titles=list(d.get("section_titles") or []),
            routing_text=str(d.get("routing_text", "")),
        )


@dataclass
class ManualDocCatalog:
    docs: list[ManualDocProfile] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]

    def _build_bm25(self) -> None:
        if not self.docs:
            self._bm25 = None
            return
        from rag.index_store import _tokenize_zh

        tokenized = [_tokenize_zh(d.routing_text) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized)

    def score_docs(self, query: str) -> dict[str, float]:
        if not self.docs or not self._bm25:
            return {}
        from rag.index_store import _tokenize_zh

        scores = self._bm25.get_scores(_tokenize_zh(query))
        return {d.doc_id: float(s) for d, s in zip(self.docs, scores)}

    def select_doc_ids(
        self,
        query: str,
        *,
        top_k: int = 2,
        min_score_ratio: float = 0.35,
    ) -> list[str]:
        """
        返回应参与块检索的 doc_id 列表（1～top_k 本）。
        仅 1 本手册时直接返回该 doc_id。
        """
        if not self.docs:
            return []
        if len(self.docs) == 1:
            return [self.docs[0].doc_id]

        ranked = sorted(self.score_docs(query).items(), key=lambda x: -x[1])
        if not ranked:
            return [d.doc_id for d in self.docs[:top_k]]

        best_score = ranked[0][1]
        if best_score <= 0:
            return [d.doc_id for d in self.docs[:top_k]]

        selected: list[str] = []
        for doc_id, score in ranked:
            if len(selected) >= top_k:
                break
            if not selected or score >= best_score * min_score_ratio:
                selected.append(doc_id)
        return selected or [ranked[0][0]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ManualDocCatalog":
        cat = cls()
        if not path.is_file():
            return cat
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cat.docs.append(ManualDocProfile.from_dict(json.loads(line)))
        cat._build_bm25()
        return cat


def build_manual_catalog(
    children: list[ChunkRecord],
    parents: list[ChunkRecord] | None = None,
    *,
    max_section_titles: int = 80,
) -> ManualDocCatalog:
    """从子块/父块 meta 聚合每本手册的路由用文本。"""
    by_doc: dict[str, dict] = {}

    for c in children:
        if c.source != "manual":
            continue
        did = str(c.meta.get("doc_id") or "").strip()
        if not did:
            continue
        slot = by_doc.setdefault(
            did,
            {
                "file_name": str(c.meta.get("file") or ""),
                "sections": [],
                "count": 0,
            },
        )
        slot["count"] += 1
        cp = (c.meta.get("chapter_path") or c.meta.get("section_title") or "").strip()
        if cp and cp not in slot["sections"]:
            slot["sections"].append(cp)

    for p in parents or []:
        if p.source != "manual":
            continue
        did = str(p.meta.get("doc_id") or "").strip()
        if not did or did not in by_doc:
            continue
        title = (p.meta.get("section_title") or "").strip()
        if title and title not in by_doc[did]["sections"]:
            by_doc[did]["sections"].append(title)

    profiles: list[ManualDocProfile] = []
    for doc_id, info in sorted(by_doc.items()):
        sections = info["sections"][:max_section_titles]
        title = doc_id
        for sec in sections:
            if sec and not sec.startswith("_"):
                title = sec.split(" > ")[-1] if " > " in sec else sec
                break
        routing_parts = [f"手册:{doc_id}", f"标题:{title}"]
        if info["file_name"]:
            routing_parts.append(f"文件:{info['file_name']}")
        if sections:
            routing_parts.append("章节:" + " | ".join(sections[:40]))
        profile = ManualDocProfile(
            doc_id=doc_id,
            title=title,
            file_name=info["file_name"],
            chunk_count=info["count"],
            section_titles=sections,
            routing_text="\n".join(routing_parts),
        )
        profiles.append(profile)

    cat = ManualDocCatalog(docs=profiles)
    cat._build_bm25()
    return cat


def infer_title_from_txt(path: Path) -> str:
    """从手册 txt 首屏 # 标题推断显示名（建库可选）。"""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return path.stem
    titles = _TITLE_RE.findall(head)
    titles = [t.strip() for t in titles if t.strip() and "目录" not in t]
    return " ".join(titles[:2]) if titles else path.stem
