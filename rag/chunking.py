"""手册文本递归分块；MinerU 输出可先合并为 .txt 再放入目录。"""

from __future__ import annotations

import re
from pathlib import Path

from rag.schemas import ChunkRecord


_HEADER_NOISE = re.compile(
    r"(版权所有|Copyright|第\s*\d+\s*页|共\s*\d+\s*页)",
    re.IGNORECASE,
)


def strip_header_footer_noise(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _HEADER_NOISE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _split_oversized(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return out


def chunk_manual_text(
    text: str,
    doc_id: str,
    chapter_path: str = "",
    page_range: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> list[ChunkRecord]:
    text = strip_header_footer_noise(text)
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n+", text)
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
                pieces.extend(_split_oversized(buf, chunk_size, chunk_overlap))
            if len(p) <= chunk_size:
                buf = p
            else:
                pieces.extend(_split_oversized(p, chunk_size, chunk_overlap))
                buf = ""
    if buf:
        pieces.extend(_split_oversized(buf, chunk_size, chunk_overlap))

    records: list[ChunkRecord] = []
    for i, t in enumerate(pieces):
        if not t.strip():
            continue
        cid = f"{doc_id}#{i:05d}"
        records.append(
            ChunkRecord(
                chunk_id=cid,
                source="manual",
                text=t,
                meta={
                    "doc_id": doc_id,
                    "chapter_path": chapter_path,
                    "page_range": page_range,
                    "chunk_index": i,
                },
            )
        )
    return records


def load_manual_from_plain_file(path: Path, doc_id: str | None = None, **kwargs) -> list[ChunkRecord]:
    text = path.read_text(encoding="utf-8", errors="replace")
    did = doc_id or path.stem
    return chunk_manual_text(text, doc_id=did, **kwargs)


def load_manual_from_directory(
    dir_path: Path,
    glob: str = "*.txt",
    **kwargs,
) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    for fp in sorted(dir_path.glob(glob)):
        if fp.is_file():
            sub = load_manual_from_plain_file(fp, doc_id=fp.stem, **kwargs)
            for c in sub:
                c.meta["file"] = str(fp.name)
            all_chunks.extend(sub)
    return all_chunks


def _nonempty_txt(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def resolve_txt_for_pdf(
    pdf: Path,
    manual_dir: Path,
    manual_txt_dir: Path,
) -> Path | None:
    """同目录旁挂 .txt 优先，其次 manual_txt_dir 下同名 .txt。"""
    side = manual_dir / f"{pdf.stem}.txt"
    exported = manual_txt_dir / f"{pdf.stem}.txt"
    if _nonempty_txt(side):
        return side
    if _nonempty_txt(exported):
        return exported
    return None


def collect_manual_txt_sources(
    manual_file: Path | None,
    manual_dir: Path | None,
    manual_txt_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    构建索引前的手册 TXT 清单。

    返回 (已就绪的 txt 路径列表, 仍需 PDF→TXT 的 pdf 路径列表)。
    目录中无 PDF 时，行为等同扫描该目录下所有 *.txt。
    """
    if manual_file is not None:
        mf = manual_file.resolve()
        if mf.suffix.lower() != ".pdf":
            return [mf], []
        side = mf.parent / f"{mf.stem}.txt"
        out = manual_txt_dir / f"{mf.stem}.txt"
        if _nonempty_txt(side):
            return [side], []
        if _nonempty_txt(out):
            return [out], []
        return [], [mf]

    if manual_dir is not None:
        md = manual_dir.resolve()
        pdfs = sorted(md.glob("*.pdf")) + sorted(md.glob("*.PDF"))
        if not pdfs:
            txts = sorted(p for p in md.glob("*.txt") if p.is_file())
            return txts, []

        ready: list[Path] = []
        need_pdf: list[Path] = []
        stems_pdf = {p.stem for p in pdfs}
        for pdf in pdfs:
            got = resolve_txt_for_pdf(pdf, md, manual_txt_dir)
            if got is not None:
                ready.append(got)
            else:
                need_pdf.append(pdf)
        for txt in sorted(md.glob("*.txt")):
            if txt.stem not in stems_pdf and txt.is_file():
                ready.append(txt)
        return ready, need_pdf

    return [], []


def load_manual_from_txt_paths(paths: list[Path], **kwargs) -> list[ChunkRecord]:
    """对已解析的一组 .txt 路径分块（去重、排序）。"""
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in sorted(paths, key=lambda x: (str(x.resolve()), x.name)):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    all_chunks: list[ChunkRecord] = []
    for fp in uniq:
        sub = load_manual_from_plain_file(fp, doc_id=fp.stem, **kwargs)
        for c in sub:
            c.meta["file"] = str(fp.name)
        all_chunks.extend(sub)
    return all_chunks
