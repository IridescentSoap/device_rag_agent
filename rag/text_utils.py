"""手册文本预处理与定长切分工具。"""

from __future__ import annotations

import re

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


def split_oversized(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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
