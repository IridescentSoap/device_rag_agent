"""请求监控：写入 logs/query_log.jsonl。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "query_log.jsonl"

_metrics: dict[str, int] = {
    "total_requests": 0,
    "need_human_confirm": 0,
}


def log_trace(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _metrics["total_requests"] += 1
    if record.get("need_human_confirm"):
        _metrics["need_human_confirm"] += 1


def get_metrics() -> dict[str, Any]:
    return {
        "total_requests": _metrics["total_requests"],
        "need_human_confirm": _metrics["need_human_confirm"],
        "log_path": str(LOG_PATH),
    }
