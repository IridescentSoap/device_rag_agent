"""检索后上下文扩展：手册父块/同节邻块，日志原因处置。"""

from __future__ import annotations

from rag.config import (
    MANUAL_EXPAND_USE_PARENT,
    MANUAL_NEIGHBOR_MAX_CHARS,
    MANUAL_PARENT_MAX_CHARS,
)
from rag.index_store import DualIndexStore
from rag.prompts import log_chunk_body_for_prompt
from rag.schemas import ChunkRecord


def _find_manual_child(
    store: DualIndexStore,
    *,
    doc_id: str | None = None,
    parent_id: str | None = None,
    chunk_index: int | None = None,
    chunk_index_in_section: int | None = None,
) -> ChunkRecord | None:
    for c in store.manual.chunks:
        if parent_id and c.meta.get("parent_id") != parent_id:
            continue
        if doc_id and c.meta.get("doc_id") != doc_id:
            continue
        if chunk_index_in_section is not None:
            if c.meta.get("chunk_index_in_section") == chunk_index_in_section:
                return c
        elif chunk_index is not None and c.meta.get("chunk_index") == chunk_index:
            return c
    return None


def neighbor_text_in_section(store: DualIndexStore, chunk: ChunkRecord) -> str:
    """同 parent_id 内 chunk_index_in_section ±1。"""
    parent_id = chunk.meta.get("parent_id")
    idx_in_sec = chunk.meta.get("chunk_index_in_section")
    if not parent_id or idx_in_sec is None:
        return neighbor_text_in_doc(store, chunk)

    extra: list[str] = []
    for delta in (-1, 1):
        nb = _find_manual_child(
            store,
            parent_id=parent_id,
            chunk_index_in_section=idx_in_sec + delta,
        )
        if nb:
            extra.append(nb.text[:MANUAL_NEIGHBOR_MAX_CHARS])
    return "\n".join(extra)


def neighbor_text_in_doc(store: DualIndexStore, chunk: ChunkRecord) -> str:
    """全书顺序 chunk_index ±1（兼容旧索引）。"""
    doc_id = chunk.meta.get("doc_id")
    idx = chunk.meta.get("chunk_index")
    if doc_id is None or idx is None:
        return ""
    extra: list[str] = []
    for delta in (-1, 1):
        nb = _find_manual_child(store, doc_id=doc_id, chunk_index=idx + delta)
        if nb:
            extra.append(nb.text[:MANUAL_NEIGHBOR_MAX_CHARS])
    return "\n".join(extra)


def parent_text_for_chunk(store: DualIndexStore, chunk: ChunkRecord) -> str:
    parent_id = chunk.meta.get("parent_id")
    if not parent_id:
        return ""
    parent = store.manual.parents.get(parent_id)
    if not parent:
        return ""
    text = parent.text.strip()
    if len(text) > MANUAL_PARENT_MAX_CHARS:
        return text[:MANUAL_PARENT_MAX_CHARS] + "\n...[章节内容已截断]..."
    return text


def manual_body_for_prompt(store: DualIndexStore, chunk: ChunkRecord) -> str:
    """手册生成正文：优先父块全文，否则子块 + 同节/全书邻块。"""
    chapter = (chunk.meta.get("chapter_path") or chunk.meta.get("section_title") or "").strip()
    header_bits = []
    if chapter:
        header_bits.append(f"[章节]{chapter}")

    if MANUAL_EXPAND_USE_PARENT and chunk.meta.get("parent_id"):
        parent_body = parent_text_for_chunk(store, chunk)
        if parent_body:
            excerpt = chunk.text.strip()
            if len(excerpt) > 800:
                excerpt = excerpt[:800] + "..."
            parts = []
            if header_bits:
                parts.append("\n".join(header_bits))
            parts.append(f"[命中子块摘录]\n{excerpt}")
            parts.append(f"[所属章节全文]\n{parent_body}")
            return "\n\n".join(parts)

    body = chunk.text
    if header_bits:
        body = "\n".join(header_bits) + "\n\n" + body
    neighbor = neighbor_text_in_section(store, chunk)
    if neighbor:
        body = f"{body}\n\n[同节相邻片段补充]\n{neighbor}"
    return body


def expand_context(store: DualIndexStore, chunks: list[ChunkRecord]) -> list[str]:
    blocks: list[str] = []
    for c in chunks:
        header = f"[{c.source}] chunk_id={c.chunk_id}"
        if c.meta.get("case_id"):
            header += f" case_id={c.meta.get('case_id')}"
        if c.meta.get("page_range"):
            header += f" page={c.meta.get('page_range')}"
        if c.meta.get("chapter_path"):
            header += f" chapter={c.meta.get('chapter_path')}"

        if c.source == "log":
            body = log_chunk_body_for_prompt(c)
        else:
            body = manual_body_for_prompt(store, c)
        blocks.append(f"{header}\n{body}")
    return blocks
