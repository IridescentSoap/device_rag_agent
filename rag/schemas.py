from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SourceType = Literal["manual", "log"]


@dataclass
class ChunkRecord:
    """单条可检索单元。"""

    chunk_id: str
    source: SourceType
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ChunkRecord":
        return ChunkRecord(
            chunk_id=d["chunk_id"],
            source=d["source"],
            text=d["text"],
            meta=d.get("meta") or {},
        )
