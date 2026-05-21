"""
FastAPI 服务：Agentic RAG /ask

启动：
  uvicorn api.server:app --host 0.0.0.0 --port 8000
或：
  python -m uvicorn api.server:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 保证项目根在 path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.monitor import get_metrics
from agent.workflow import AgentWorkflow

app = FastAPI(
    title="Device RAG Agent API",
    description="空管设备运维 Agentic RAG",
    version="0.1.0",
)

_workflow: AgentWorkflow | None = None


def get_executor() -> AgentWorkflow:
    """返回 Agent 工作流（LangGraph 或 executor 回退）。"""
    global _workflow
    if _workflow is None:
        _workflow = AgentWorkflow()
    return _workflow


class HistoryTurn(BaseModel):
    role: str = "user"
    content: str = ""


class AskRequest(BaseModel):
    query: str
    history: list[HistoryTurn] = Field(default_factory=list)
    skip_llm: bool = False
    fast_mode: bool = False


class AskResponse(BaseModel):
    answer: str
    route: str
    rewritten_query: str
    tools_used: list[str]
    citations: list[str]
    confidence: float
    need_human_confirm: bool
    latency_ms: int
    fast_mode: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return get_metrics()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    history = [h.model_dump() for h in req.history]
    resp = get_executor().run(
        req.query, history, skip_llm=req.skip_llm, fast_mode=req.fast_mode
    )
    d = resp.to_dict()
    return AskResponse(
        answer=d["answer"],
        route=d["route"],
        rewritten_query=d["rewritten_query"],
        tools_used=d["tools_used"],
        citations=d["citations"],
        confidence=d["confidence"],
        need_human_confirm=d["need_human_confirm"],
        latency_ms=d["latency_ms"],
        fast_mode=d.get("fast_mode", False),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
