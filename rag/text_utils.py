"""手册文本预处理与定长切分工具。"""

from __future__ import annotations

import re

try:
    from llama_index.core.node_parser import SentenceSplitter as _SentenceSplitter
except ImportError:
    _SentenceSplitter = None  # type: ignore[misc, assignment]

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


def _split_oversized_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """字符滑动窗口切分（SentenceSplitter 不可用时的回退）。"""
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


def split_oversized(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    超长文本切分：优先按句子边界（LlamaIndex SentenceSplitter），否则回退字符窗。
    仅用于节内段落仍超过 chunk_size 的场景；段落合并逻辑不变。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if _SentenceSplitter is not None:
        splitter = _SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
        if pieces:
            return pieces
    return _split_oversized_by_chars(text, chunk_size, chunk_overlap)
