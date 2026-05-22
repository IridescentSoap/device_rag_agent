"""手册结构化分块：Markdown 章节解析、目录过滤、父子块。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.text_utils import split_oversized, strip_header_footer_noise
from rag.schemas import ChunkRecord

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# MinerU 手册常见：正文标题均为 ##，层级在「1. / 1.1. / 2.1.1.1.」编号中
_OUTLINE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[、．.]|\s)")
_TOC_TITLE_RE = re.compile(r"(目\s*录|^目录$|修订记录|版本说明|版本历史|前\s*言\s*$)")
_TOC_LINE_RE = re.compile(
    r"^\s*\d+(?:\.\d+)+\.?\s+.+\.{2,}",
)
_REVISION_TABLE_RE = re.compile(r"<table>.*?(版本|V\d+\.\d+)", re.DOTALL)
# MinerU 误标为 ## 的节内列表/图注/要点，不作为章节标题
_DECORATIVE_HEADING_RE = re.compile(r"^(图\s|⚫|■|●)")
_TOP_CHAPTER_HINT_RE = re.compile(r"(前言|功能描述|外部交互|附录|系统总体|系统说明)")


@dataclass
class Section:
    """一个 Markdown 标题节（含标题栈与正文）。"""

    section_index: int
    title: str
    chapter_path: str
    body: str
    heading_level: int


@dataclass
class ManualChunkBundle:
    """结构化分块产物：子块用于检索，父块用于生成扩展。"""

    children: list[ChunkRecord] = field(default_factory=list)
    parents: list[ChunkRecord] = field(default_factory=list)


def _slug_section(section_index: int) -> str:
    return f"sec_{section_index:04d}"


def _outline_number_parts(title: str) -> list[str] | None:
    m = _OUTLINE_NUM_RE.match(title.strip())
    if not m:
        return None
    return m.group(1).split(".")


def _max_level1_chapter_num(heading_stack: list[tuple[int, str]]) -> int:
    best = 0
    for lev, t in heading_stack:
        if lev != 1:
            continue
        parts = _outline_number_parts(t)
        if parts and len(parts) == 1:
            best = max(best, int(parts[0]))
    return best


def _is_decorative_heading(title: str) -> bool:
    """图注、⚫ 要点等：保留为正文，不切 Section。"""
    return bool(_DECORATIVE_HEADING_RE.match(title.strip()))


def _resolve_heading_level(
    title: str,
    hash_level: int,
    heading_stack: list[tuple[int, str]],
) -> int:
    """
    推断标题层级：优先用编号深度（1.1.2 → 3），否则回退 # 个数或栈顶 +1。

    适配 MinerU 导出手册（正文多为 ## + 数字编号，而非 ###/####）。
    单段编号（## 1. / ## 3.）在深栈下多为节内列表，仅在新章（如 3. 外部交互）时为 level 1。
    """
    parts = _outline_number_parts(title)
    if parts:
        if len(parts) >= 2:
            return max(1, min(6, len(parts)))
        n = int(parts[0])
        max_ch = _max_level1_chapter_num(heading_stack)
        if _TOP_CHAPTER_HINT_RE.search(title) and (not heading_stack or n > max_ch):
            return 1
        if heading_stack:
            top_level = heading_stack[-1][0]
            if top_level >= 3:
                return min(6, top_level + 1)
            if top_level <= 2:
                return 1
        return 1
    if hash_level == 1:
        return 1
    if heading_stack:
        return max(2, min(6, heading_stack[-1][0] + 1))
    return max(hash_level, 2)


def parse_markdown_sections(text: str) -> list[Section]:
    """按 Markdown 标题切分为节；编号标题用 1.1.1 深度建 chapter_path。"""
    lines = text.splitlines()
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    sec_idx = 0
    current_title = ""
    current_level = 0

    def chapter_path() -> str:
        return " > ".join(t for _, t in heading_stack) if heading_stack else ""

    def flush() -> None:
        nonlocal sec_idx, buf, current_title, current_level
        body = "\n".join(buf).strip()
        if not body and not current_title:
            buf = []
            return
        sections.append(
            Section(
                section_index=sec_idx,
                title=current_title or "_preamble",
                chapter_path=chapter_path() or current_title or "_preamble",
                body=body,
                heading_level=current_level,
            )
        )
        sec_idx += 1
        buf = []

    for line in lines:
        m = _HEADER_RE.match(line.strip())
        if m:
            hash_level = len(m.group(1))
            title = m.group(2).strip()
            if hash_level >= 2 and _is_decorative_heading(title):
                buf.append(line)
                continue
            flush()
            level = _resolve_heading_level(title, hash_level, heading_stack)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_title = title
            current_level = level
        else:
            buf.append(line)
    flush()
    return sections


def is_toc_or_noise_section(section: Section) -> bool:
    """目录页、修订表、纯目录条目等不参与检索索引。"""
    title = (section.title or "").strip()
    body = (section.body or "").strip()
    if not body and title in ("_preamble", ""):
        return True
    if _TOC_TITLE_RE.search(title):
        return True
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return True
    toc_hits = sum(
        1
        for ln in lines
        if _TOC_LINE_RE.search(ln) or "......" in ln or "…" in ln
    )
    if len(lines) >= 4 and toc_hits / len(lines) >= 0.35:
        return True
    if title == "_preamble" and _REVISION_TABLE_RE.search(body):
        return True
    if "<table>" in body and len(body) < 4000:
        if _REVISION_TABLE_RE.search(body) and toc_hits == 0:
            return True
    return False


def _chunk_section_body(
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    paragraphs = re.split(r"\n\s*\n+", body)
    pieces: list[str] = []
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                pieces.extend(split_oversized(buf, chunk_size, chunk_overlap))
            if len(p) <= chunk_size:
                buf = p
            else:
                pieces.extend(split_oversized(p, chunk_size, chunk_overlap))
                buf = ""
    if buf:
        pieces.extend(split_oversized(buf, chunk_size, chunk_overlap))
    return [t for t in pieces if t.strip()]


def chunk_manual_text_structured(
    text: str,
    doc_id: str,
    *,
    page_range: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    filter_toc: bool = True,
) -> ManualChunkBundle:
    """
    章节分块 → 目录过滤 → 节内子块 + 节级父块。

    - child：写入向量/BM25 索引，chunk_id 仍为 {doc_id}#{idx:05d}
    - parent：仅落盘 parents.jsonl，供生成阶段扩展
    """
    text = strip_header_footer_noise(text)
    if not text.strip():
        return ManualChunkBundle()

    sections = parse_markdown_sections(text)
    children: list[ChunkRecord] = []
    parents: list[ChunkRecord] = []
    global_idx = 0

    for section in sections:
        if filter_toc and is_toc_or_noise_section(section):
            continue

        sec_slug = _slug_section(section.section_index)
        parent_id = f"{doc_id}#{sec_slug}"
        full_section_text = section.body.strip()
        if section.title and section.title != "_preamble":
            full_section_text = f"## {section.title}\n\n{full_section_text}"

        child_texts = _chunk_section_body(full_section_text, chunk_size, chunk_overlap)
        if not child_texts and full_section_text.strip():
            child_texts = [full_section_text.strip()]

        child_ids: list[str] = []
        for i, t in enumerate(child_texts):
            cid = f"{doc_id}#{global_idx:05d}"
            child_ids.append(cid)
            children.append(
                ChunkRecord(
                    chunk_id=cid,
                    source="manual",
                    text=t,
                    meta={
                        "doc_id": doc_id,
                        "chunk_role": "child",
                        "parent_id": parent_id,
                        "chapter_path": section.chapter_path,
                        "section_title": section.title,
                        "section_index": section.section_index,
                        "chunk_index": global_idx,
                        "chunk_index_in_section": i,
                        "page_range": page_range,
                        "indexable": True,
                    },
                )
            )
            global_idx += 1

        if not full_section_text.strip():
            continue

        parents.append(
            ChunkRecord(
                chunk_id=parent_id,
                source="manual",
                text=full_section_text,
                meta={
                    "doc_id": doc_id,
                    "chunk_role": "parent",
                    "parent_id": parent_id,
                    "chapter_path": section.chapter_path,
                    "section_title": section.title,
                    "section_index": section.section_index,
                    "child_ids": child_ids,
                    "page_range": page_range,
                    "indexable": False,
                },
            )
        )

    return ManualChunkBundle(children=children, parents=parents)
