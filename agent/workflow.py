"""
Agent 工作流：优先 LangGraph，不可用时回退到 agent.executor.AgentExecutor。

Query Decomposition 流水线：
  rewrite_query -> plan_query -> decompose_query -> retrieve (per sub_query)
  -> judge_evidence -> [supplement_search -> judge_evidence]* -> generate_answer -> log_trace
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from agent.context import rewrite_query_with_meta
from agent.decompose import decompose_query
from agent.evidence import judge_evidence
from agent.executor import AgentExecutor
from agent.hits import merge_hits
from agent.planner import plan_query
from agent.retrieve import retrieve_sub_queries
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
    rewrite: dict[str, Any]
    plan: dict[str, Any]
    decompose: dict[str, Any]
    hits: list[RerankHit]
    pipeline_route: str
    evidence: dict[str, Any]
    answer: str
    citations: list[str]
    early_exit: bool
    supplement_rounds: int
    max_supplement_rounds: int
    supplement_queries: list[str]
    use_llm_planner: bool | None
    use_llm_decompose: bool | None
    use_llm_rewrite: bool | None


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
        max_supplement_rounds: int = 1,
        use_llm_planner: bool | None = None,
        use_llm_decompose: bool | None = None,
        use_llm_rewrite: bool | None = None,
    ) -> AgentResponse:
        if self._graph is None:
            return self._fallback.run(
                query,
                history,
                skip_llm=skip_llm,
                fast_mode=fast_mode,
                max_supplement_rounds=max_supplement_rounds,
                use_llm_planner=use_llm_planner,
                use_llm_decompose=use_llm_decompose,
                use_llm_rewrite=use_llm_rewrite,
            )
        return self._run_langgraph(
            query,
            history,
            skip_llm=skip_llm,
            fast_mode=fast_mode,
            max_supplement_rounds=max_supplement_rounds,
            use_llm_planner=use_llm_planner,
            use_llm_decompose=use_llm_decompose,
            use_llm_rewrite=use_llm_rewrite,
        )

    def _run_langgraph(
        self,
        query: str,
        history: list[dict[str, Any]] | None,
        *,
        skip_llm: bool,
        fast_mode: bool | None,
        max_supplement_rounds: int,
        use_llm_planner: bool | None,
        use_llm_decompose: bool | None,
        use_llm_rewrite: bool | None,
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
                "max_supplement_rounds": max_supplement_rounds,
                "supplement_rounds": 0,
                "supplement_queries": [],
                "use_llm_planner": use_llm_planner,
                "use_llm_decompose": use_llm_decompose,
                "use_llm_rewrite": use_llm_rewrite,
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
            supplement_rounds=int(state.get("supplement_rounds") or 0),
        )
        AgentExecutor._log(query, resp, history=history, tools=tools)
        return resp

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(WorkflowState)
        graph.add_node("rewrite_query", self._node_rewrite)
        graph.add_node("plan_query", self._node_plan)
        graph.add_node("decompose_query", self._node_decompose)
        graph.add_node("insufficient", self._node_insufficient)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("judge_evidence", self._node_judge)
        graph.add_node("supplement_search", self._node_supplement_search)
        graph.add_node("generate_answer", self._node_generate)

        graph.add_edge(START, "rewrite_query")
        graph.add_edge("rewrite_query", "plan_query")
        graph.add_conditional_edges(
            "plan_query",
            self._route_after_plan,
            {"insufficient": "insufficient", "decompose": "decompose_query"},
        )
        graph.add_edge("insufficient", END)
        graph.add_edge("decompose_query", "retrieve")
        graph.add_edge("retrieve", "judge_evidence")
        graph.add_conditional_edges(
            "judge_evidence",
            self._route_after_judge,
            {"supplement_search": "supplement_search", "generate_answer": "generate_answer"},
        )
        graph.add_edge("supplement_search", "judge_evidence")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    @staticmethod
    def _route_after_plan(state: WorkflowState) -> str:
        plan = state.get("plan") or {}
        if plan.get("route") == "insufficient_evidence":
            return "insufficient"
        return "decompose"

    @staticmethod
    def _route_after_judge(state: WorkflowState) -> str:
        ev = state.get("evidence") or {}
        rounds = int(state.get("supplement_rounds") or 0)
        max_rounds = int(state.get("max_supplement_rounds") or 0)
        missing = ev.get("missing_aspects") or []
        supplement_queries = ev.get("supplement_queries") or []
        if (
            (ev.get("need_human_confirm") or missing)
            and supplement_queries
            and rounds < max_rounds
        ):
            return "supplement_search"
        return "generate_answer"

    def _node_rewrite(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        rewrite = rewrite_query_with_meta(
            state["query"],
            state.get("history"),
            use_llm_rewrite=state.get("use_llm_rewrite"),
        )
        tools_used.append("rewrite_query")
        if rewrite.rewriter_type == "llm":
            tools_used.append("rewrite_query_llm")
        elif rewrite.rewriter_type == "rule_fallback":
            tools_used.append("rewrite_query_rule_fallback")
        return {
            "rewritten_query": rewrite.query,
            "rewrite": rewrite.to_dict(),
            "tools_used": tools_used,
        }

    def _node_plan(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        plan = plan_query(
            state["query"],
            state.get("history"),
            rewritten_query=state.get("rewritten_query"),
            use_llm_planner=state.get("use_llm_planner"),
        )
        tools_used.append("plan_query")
        return {"plan": plan.to_dict(), "tools_used": tools_used}

    def _node_decompose(self, state: WorkflowState) -> WorkflowState:
        tools_used = list(state.get("tools_used") or [])
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        decompose = decompose_query(
            plan.rewritten_query,
            route=plan.route,
            needs_manual=plan.needs_manual,
            needs_log=plan.needs_log,
            use_llm_decompose=state.get("use_llm_decompose"),
        )
        tools_used.append("decompose_query")
        plan_dict = dict(state["plan"])
        plan_dict["sub_queries"] = [sq.text for sq in decompose.sub_queries]
        plan_dict["decompose"] = decompose.to_dict()
        if state.get("rewrite"):
            plan_dict["rewrite"] = state["rewrite"]
        return {
            "plan": plan_dict,
            "decompose": decompose.to_dict(),
            "tools_used": tools_used,
        }

    def _node_insufficient(self, state: WorkflowState) -> WorkflowState:
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
        decompose_dict = state.get("decompose") or (state.get("plan") or {}).get(
            "decompose"
        )
        from agent.decompose import SubQuery

        sub_queries = [
            SubQuery(
                text=sq["text"],
                aspect=sq.get("aspect", "general"),  # type: ignore[arg-type]
                prefer_manual=bool(sq.get("prefer_manual")),
                prefer_log=bool(sq.get("prefer_log")),
            )
            for sq in (decompose_dict or {}).get("sub_queries") or []
        ]
        hits, pipeline_route = retrieve_sub_queries(tools, plan, sub_queries, tools_used)
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
            query=plan.rewritten_query,
            route=plan.route,
        )
        tools_used.append("judge_evidence")
        return {
            "evidence": ev.to_dict(),
            "tools_used": tools_used,
            "supplement_queries": ev.supplement_queries,
        }

    def _node_supplement_search(self, state: WorkflowState) -> WorkflowState:
        tools = self._current_tools
        assert tools is not None
        tools_used = list(state.get("tools_used") or [])
        plan = _plan_from_dict(state["plan"])  # type: ignore[arg-type]
        ev = state.get("evidence") or {}
        supplement_queries = list(
            state.get("supplement_queries") or ev.get("supplement_queries") or []
        )
        hits = list(state.get("hits") or [])
        from agent.decompose import SubQuery

        sub_queries = [SubQuery(text=sq) for sq in supplement_queries]
        extra, _ = retrieve_sub_queries(tools, plan, sub_queries, tools_used)
        merged = merge_hits(hits, extra)
        rounds = int(state.get("supplement_rounds") or 0) + 1
        return {
            "hits": merged,
            "tools_used": tools_used,
            "supplement_rounds": rounds,
            "supplement_queries": supplement_queries,
        }

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
        sub_query_texts = list((state.get("plan") or {}).get("sub_queries") or [])

        if skip_llm or not hits:
            if hits:
                answer = (
                    f"已检索到 {len(hits)} 条相关片段（路由={pipeline_route}，"
                    f"原子子问题={len(sub_query_texts)}），"
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
                sub_queries=sub_query_texts if len(sub_query_texts) > 1 else None,
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
