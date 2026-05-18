"""从运维 CSV 构建日志 ChunkRecord（兼容 falut_* / fault_* 字段名）。"""

from __future__ import annotations

import pandas as pd

from rag.schemas import ChunkRecord


def _pick(row: pd.Series, *names: str) -> str:
    for n in names:
        if n not in row.index:
            continue
        v = row[n]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def row_to_log_text(row: pd.Series) -> str:
    system = _pick(row, "system")
    phen = _pick(row, "falut_description", "fault_description")
    inf = _pick(row, "falut_influence", "fault_influence")
    parts = []
    if system:
        parts.append(f"[system]{system}")
    if phen:
        parts.append(f"[phenomenon]{phen}")
    if inf:
        parts.append(f"[impact]{inf}")
    return " ".join(parts)


def dataframe_to_log_chunks(df: pd.DataFrame) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for i in range(len(df)):
        row = df.iloc[i]
        text = row_to_log_text(row)
        if not text.strip():
            continue
        # 优先使用 CSV 中已有 case_id，缺失时回退为默认 ID
        case_id = _pick(row, "case_id")
        cid = case_id if case_id else f"log_case_{i}"
        chunks.append(
            ChunkRecord(
                chunk_id=cid,
                source="log",
                text=text,
                meta={
                    "row_index": i,
                    "case_id": cid,
                    "system": _pick(row, "system"),
                    "device": _pick(row, "device"),
                    # 供生成阶段拼上下文，无需再读 CSV（兼容列名拼写）
                    "cause": _pick(row, "falut_cause", "fault_cause"),
                    "solution": _pick(row, "falut_solution", "fault_solution"),
                },
            )
        )
    return chunks


def load_logs_from_csv(path: str) -> list[ChunkRecord]:
    df = pd.read_csv(path)
    return dataframe_to_log_chunks(df)
