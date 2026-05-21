"""
Agent 工作流：优先 LangGraph，不可用时回退到 agent.executor.AgentExecutor。

流程与 executor 一致：
  rewrite_query -> plan_query -> retrieve -> judge_evidence -> generate_answer -> log_trace
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from agent.context import rewrite_query
from agent.evidence import judge_evidence
from agent.executor import AgentExecutor
from agent.fast_mode import resolve_fast_mode
from agent.planner import plan_query
from agent.state import AgentResponse, PlanResult
from agent.tools import RagTools
from rag.rerank import RerankHit

_LANGGRAPH_AVAILABLE: bool | None = None


def langgraph_available() -> bool:
    """是否已安装并可导入 LangGraph。"""
    global _LANGGRAPH_AVAILABLE
    if _LANGGRAPH_AVAILABLE is not None:
        return _LANGGRAPH_AVAILABLE
    try:
        from langgraph.graph import END, START, StateGraph  # noqa: F401

        _LANGGRAPH_AVAILABLE = True
    except Exception:
        _LANGGRAPH_AVAILABLE = False
    return _LANGGRAPH_AVAILABLE


class WorkflowState(TypedDict, total=False):
    query: str
    history: list[dict[str, Any]]
    skip_llm: bool
    fast_mode: bool | None
    t0: float
    tools_used: list[str]
    rewritten_query: str
    plan: dict[str, Any]
    hits: list[RerankHit]
    pipeline_route: str
    evidence: dict[str, Any]
    answer: str
    citations: list[str]
    early_exit: bool


def _plan_from_dict(d: dict[str, Any]) -> PlanResult:
    return PlanResult(
        route=d["route"],  # type: ignore[arg-type]
        rewritten_query=d["rewritten_query"],
        sub_queries=list(d.get("sub_queries") or []),
        needs_manual=bool(d.get("needs_manual")),
        needs_log=bool(d.get("needs_log")),
        confidence=float(d.get("confidence", 0.8)),
    )


class AgentWorkflow:
    """
    统一 Agent 入口：LangGraph 可用时走图编排，否则委托 AgentExecutor。
    """

    def __init__(self, tools: RagTools | None = None):
        self._tools = tools
        self._fallback = AgentExecutor(tools=tools)
        self._graph = None
        self._current_tools: RagTools | None = None
        if langgraph_available():
            self._graph = self._build_graph()

    @property
    def backend(self) -> Literal["langgraph", "executor"]:
        return "langgraph" if self._graph is not None else "executor"

    def run(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
        *,
        skip_llm: bool = False,
        fast_mode: bool | None = None,
    ) -> AgentResponse:
        if self._graph is None:
            return self._fallback.run(
                query, history, skip_llm=skip_llm, fast_mode=fast_mode
            )
        return self._run_langgraph(
            query, history, skip_llm=skip_llm, fast_mode=fast_mode
        )

    def _run_langgraph(
        self,
        query: str,
        history: list[dict[str, Any]] | None,
        *,
        skip_llm: bool,
        fast_mode: bool | None,
    ) -> AgentResponse:
        tools = self._fallback._tools_for_run(fast_mode)
        self._current_tools = tools
        try:
            init: WorkflowState = {
                "query": query,
                "history": list(history or []),
                "skip_llm": skip_llm,
                "fast_mode": fast_mode,
                "t0": time.perf_counter(),
                "tools_used": [],
                "hits": [],
                "pipeline_route": "balanced",
                "citations": [],
                "early_exit": False,
            }
            if tools.fast_mode:
                init["tools_used"] = ["fast_mode"]
            final = self._graph.invoke(init)
            return self._state_to_response(query, history, tools, final)
        finally:
            self._current_tools = None

    def _state_to_response(
        self,
        query: str,
        history: list[dict[str, Any]] | None,
        tools: RagTools,
        state: WorkflowState,
    ) -> AgentResponse:
        t0 = float(state.get("t0") or time.perf_counter())
        plan = state.get("plan") or {}
        ev = state.get("evidence") or {}
        resp = AgentResponse(
            answer=state.get("answer") or "",
            route=str(plan.get("route", "hybrid_diagnosis")),
            rewritten_query=state.get("rewritten_query") or query,
            tools_used=list(state.get("tools_used") or []),
            citations=list(state.get("citations") or []),
            confidence=float(ev.get("confidence", 0.0)),
            need_human_confirm=bool(ev.get("need_human_confirm", False))
            or not bool(state.get("citations")),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            fast_mode=tools.fast_mode,
            plan=plan,
            evidence=ev,
        )
        AgentExecutor._log(query, resp, history=history, tools=tools)
        return resp

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(WorkflowState)
        graph.add_node("rewrite_query", self._node_rewrite)
        graph.add_node("plan_query", self._node_plan)
        graph.add_node("insufficient", self._node_insufficient)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("judge_evidence", self._node_judge)
        graph.add_node("generate_answer", self._node_generate)

        graph.add_edge(START, "rewrite_query")
        graph.add_edge("rewrite_query", "plan_query")
        graph.add_conditional_edges(
            "plan_query",
            self._route_after_plan,
            {"insufficient": "insufficient", "retrieve": "retrieve"},
        )
        graph.add_edge("insufficient", END)
        graph.add_edge("retrieve", "judge_evidence")
        graph.add_edge("judge_evidence", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    @staticmethod
    def _route_after_plan(state: WorkflowState) -> str:
        plan = state.get("plan") or {}
        if plan.get("route") == "insufficient_evidence":
            return "insufficient"
        return "retrieve"

    def _node_rewrite(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        rewritten = rewrite_query(state["query"], state.get("history"))
        tools_used.append("rewrite_query")
        return {"rewritten_query": rewritten, "tools_used": tools_used}

    def _node_plan(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        plan = plan_query(
            state["query"],
            state.get("history"),
            rewritten_query=state.get("rewritten_query"),
        )
        tools_used.append("plan_query")
        return {"plan": plan.to_dict(), "tools_used": tools_used}

    def _node_insufficient(self, state: WorkflowState) -> WorkflowState:
        tools = self._current_tools
        assert tools is not None
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        tools_used = list(state.get("tools_used") or [])
        ev = judge_evidence([], pipeline_route="balanced", plan_confidence=plan.confidence)
        answer = (
            "您的问题信息不足，请补充设备/系统名称、故障现象或想查阅的手册主题，"
            "以便检索手册与历史案例。"
        )
        return {
            "early_exit": True,
            "answer": answer,
            "citations": [],
            "evidence": ev.to_dict(),
            "tools_used": tools_used,
        }

    def _node_retrieve(self, state: WorkflowState) -> WorkflowState:
        tools = self._current_tools
        assert tools is not None
        tools_used = list(state.get("tools_used") or [])
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        q = plan.rewritten_query
        hits: list[RerankHit] = []
        pipeline_route = "balanced"

        if plan.route == "manual_query":
            hits = tools.search_manual(q)
            tools_used.append("search_manual")
            pipeline_route = "manual_heavy"
        elif plan.route == "log_case_query":
            hits = tools.search_logs(q)
            tools_used.append("search_logs")
            pipeline_route = "log_heavy"
        else:
            hits, pipeline_route = tools.hybrid_search(q)
            tools_used.append("hybrid_search")

        return {
            "hits": hits,
            "pipeline_route": pipeline_route,
            "tools_used": tools_used,
        }

    def _node_judge(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        ev = judge_evidence(
            list(state.get("hits") or []),
            pipeline_route=state.get("pipeline_route") or "balanced",
            plan_confidence=plan.confidence,
        )
        tools_used.append("judge_evidence")
        return {"evidence": ev.to_dict(), "tools_used": tools_used}

    def _node_generate(self, state: WorkflowState) -> WorkflowState:
        tools = self._current_tools
        assert tools is not None
        tools_used = list(state.get("tools_used") or [])
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        hits = list(state.get("hits") or [])
        ev = state.get("evidence") or {}
        skip_llm = bool(state.get("skip_llm"))
        pipeline_route = state.get("pipeline_route") or "balanced"
        q = plan.rewritten_query

        if skip_llm or not hits:
            if hits:
                answer = (
                    f"已检索到 {len(hits)} 条相关片段（路由={pipeline_route}），"
                    "未调用 LLM 生成。请配置 LLM_API_KEY 后重试完整回答。"
                )
            else:
                answer = (
                    "未召回到足够相关的参考资料。"
                    + ("；".join(ev.get("missing_aspects") or []))
                )
            cites = list(ev.get("citations") or [])
        else:
            answer, cites = tools.generate_answer(
                q,
                hits,
                agent_route=plan.route,
                pipeline_route=pipeline_route,
            )
            tools_used.append("generate_answer")

        return {
            "answer": answer,
            "citations": cites,
            "tools_used": tools_used,
        }


def run_agent(
    query: str,
    history: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AgentResponse:
    """与 executor.run_agent 相同签名，经 workflow 调度。"""
    return AgentWorkflow().run(query, history, **kwargs)
