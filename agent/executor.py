"""线性 Agent 执行器（无 LangGraph）。"""

from __future__ import annotations

import time
from typing import Any

from agent.context import rewrite_query
from agent.evidence import judge_evidence
from agent.monitor import log_trace
from agent.planner import plan_query
from agent.state import AgentResponse, PlanResult
from agent.fast_mode import resolve_fast_mode
from agent.tools import RagTools
from rag.rerank import RerankHit


def _merge_hits(*hit_lists: list[RerankHit]) -> list[RerankHit]:
    """按 chunk_id 去重，保留更高 score，最终按 score 降序。"""
    by_id: dict[str, RerankHit] = {}
    for batch in hit_lists:
        for h in batch:
            cid = h.chunk.chunk_id
            prev = by_id.get(cid)
            if prev is None or float(h.score) > float(prev.score):
                by_id[cid] = h
    return sorted(by_id.values(), key=lambda x: float(x.score), reverse=True)


class AgentExecutor:
    def __init__(self, tools: RagTools | None = None):
        self.tools = tools or RagTools()

    def _tools_for_run(self, fast_mode: bool | None) -> RagTools:
        use_fast = resolve_fast_mode(fast_mode)
        if self.tools.fast_mode == use_fast:
            return self.tools
        return RagTools(
            index_dir=self.tools.index_dir,
            device=self.tools.device,
            fast_mode=use_fast,
        )

    def _supplement_retrieve(
        self,
        tools: RagTools,
        plan: PlanResult,
        supplement_queries: list[str],
        tools_used: list[str],
    ) -> list[RerankHit]:
        new_hits: list[RerankHit] = []
        manual_only = plan.needs_manual and not plan.needs_log
        log_only = plan.needs_log and not plan.needs_manual

        for sq in supplement_queries:
            if manual_only:
                new_hits.extend(tools.search_manual(sq))
                tools_used.append("supplement_search_manual")
            elif log_only:
                new_hits.extend(tools.search_logs(sq))
                tools_used.append("supplement_search_logs")
            else:
                batch, _ = tools.hybrid_search(sq)
                new_hits.extend(batch)
                tools_used.append("supplement_search_hybrid")
        return new_hits

    def run(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
        *,
        skip_llm: bool = False,
        fast_mode: bool | None = None,
        max_supplement_rounds: int = 1,
        use_llm_planner: bool | None = None,
    ) -> AgentResponse:
        tools = self._tools_for_run(fast_mode)
        t0 = time.perf_counter()
        tools_used: list[str] = []
        if tools.fast_mode:
            tools_used.append("fast_mode")
        hits: list[RerankHit] = []
        pipeline_route = "balanced"
        supplement_rounds = 0

        # 1) rewrite
        rewritten = rewrite_query(query, history)
        tools_used.append("rewrite_query")

        # 2) plan
        plan = plan_query(
            query, history, rewritten_query=rewritten, use_llm_planner=use_llm_planner
        )
        tools_used.append("plan_query")

        if plan.route == "insufficient_evidence":
            ev = judge_evidence([], pipeline_route="balanced", plan_confidence=plan.confidence)
            answer = (
                "您的问题信息不足，请补充设备/系统名称、故障现象或想查阅的手册主题，"
                "以便检索手册与历史案例。"
            )
            latency = int((time.perf_counter() - t0) * 1000)
            resp = AgentResponse(
                answer=answer,
                route=plan.route,
                rewritten_query=rewritten,
                tools_used=tools_used,
                citations=[],
                confidence=ev.confidence,
                need_human_confirm=True,
                latency_ms=latency,
                fast_mode=tools.fast_mode,
                plan=plan.to_dict(),
                evidence=ev.to_dict(),
                supplement_rounds=0,
            )
            self._log(query, resp, tools=tools)
            return resp

        # 3) retrieve
        q = plan.rewritten_query
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

        # 4) judge evidence
        ev = judge_evidence(
            hits,
            pipeline_route=pipeline_route,
            plan_confidence=plan.confidence,
            query=q,
            route=plan.route,
        )
        tools_used.append("judge_evidence")

        # 4b) supplement retrieval
        if (
            max_supplement_rounds > 0
            and ev.supplement_queries
            and (ev.need_human_confirm or ev.missing_aspects)
        ):
            for _ in range(max_supplement_rounds):
                supplement_rounds += 1
                extra = self._supplement_retrieve(
                    tools, plan, ev.supplement_queries, tools_used
                )
                hits = _merge_hits(hits, extra)
                ev = judge_evidence(
                    hits,
                    pipeline_route=pipeline_route,
                    plan_confidence=plan.confidence,
                    query=q,
                    route=plan.route,
                )
                tools_used.append("judge_evidence")
                if not ev.missing_aspects and not ev.need_human_confirm:
                    break

        # 5) generate
        if skip_llm or not hits:
            if hits:
                answer = (
                    f"已检索到 {len(hits)} 条相关片段（路由={pipeline_route}），"
                    "未调用 LLM 生成。请配置 LLM_API_KEY 后重试完整回答。"
                )
            else:
                answer = (
                    "未召回到足够相关的参考资料。"
                    + ("；".join(ev.missing_aspects) if ev.missing_aspects else "")
                )
            cites = ev.citations
        else:
            answer, cites = tools.generate_answer(
                q,
                hits,
                agent_route=plan.route,
                pipeline_route=pipeline_route,
            )
            tools_used.append("generate_answer")

        latency = int((time.perf_counter() - t0) * 1000)
        resp = AgentResponse(
            answer=answer,
            route=plan.route,
            rewritten_query=rewritten,
            tools_used=tools_used,
            citations=cites,
            confidence=ev.confidence,
            need_human_confirm=ev.need_human_confirm or not cites,
            latency_ms=latency,
            fast_mode=tools.fast_mode,
            plan=plan.to_dict(),
            evidence=ev.to_dict(),
            supplement_rounds=supplement_rounds,
        )
        self._log(query, resp, history=history, tools=tools)
        return resp

    @staticmethod
    def _log(
        query: str,
        resp: AgentResponse,
        history: list | None = None,
        tools: RagTools | None = None,
    ) -> None:
        entry = {
            "query": query,
            "rewritten_query": resp.rewritten_query,
            "route": resp.route,
            "tools_used": resp.tools_used,
            "citations": resp.citations,
            "confidence": resp.confidence,
            "need_human_confirm": resp.need_human_confirm,
            "latency_ms": resp.latency_ms,
            "fast_mode": resp.fast_mode,
            "history_turns": len(history or []),
            "supplement_rounds": resp.supplement_rounds,
            "supplement_queries": (resp.evidence or {}).get("supplement_queries", []),
        }
        if tools and tools.fast_mode:
            entry["fast_config"] = tools.mode_config()
        log_trace(entry)


def run_agent(
    query: str,
    history: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AgentResponse:
    return AgentExecutor().run(query, history, **kwargs)
