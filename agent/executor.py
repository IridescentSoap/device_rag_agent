"""线性 Agent 执行器（无 LangGraph）。"""

from __future__ import annotations

import time
from typing import Any

from agent.context import rewrite_query_with_meta
from agent.decompose import decompose_query
from agent.evidence import judge_evidence
from agent.hits import merge_hits
from agent.monitor import log_trace
from agent.planner import plan_query
from agent.retrieve import retrieve_sub_queries
from agent.state import AgentResponse, PlanResult
from agent.fast_mode import resolve_fast_mode
from agent.tools import RagTools
from rag.rerank import RerankHit


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
        from agent.decompose import SubQuery

        sub_queries = [SubQuery(text=sq) for sq in supplement_queries]
        hits, _ = retrieve_sub_queries(tools, plan, sub_queries, tools_used)
        return hits

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
        tools = self._tools_for_run(fast_mode)
        t0 = time.perf_counter()
        tools_used: list[str] = []
        if tools.fast_mode:
            tools_used.append("fast_mode")
        hits: list[RerankHit] = []
        pipeline_route = "balanced"
        supplement_rounds = 0

        # 1) rewrite
        rewrite = rewrite_query_with_meta(
            query, history, use_llm_rewrite=use_llm_rewrite
        )
        rewritten = rewrite.query
        tools_used.append("rewrite_query")
        if rewrite.rewriter_type == "llm":
            tools_used.append("rewrite_query_llm")
        elif rewrite.rewriter_type == "rule_fallback":
            tools_used.append("rewrite_query_rule_fallback")

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

        # 3) decompose
        decompose = decompose_query(
            plan.rewritten_query,
            route=plan.route,
            needs_manual=plan.needs_manual,
            needs_log=plan.needs_log,
            use_llm_decompose=use_llm_decompose,
        )
        tools_used.append("decompose_query")
        plan_dict = plan.to_dict()
        plan_dict["sub_queries"] = [sq.text for sq in decompose.sub_queries]
        plan_dict["decompose"] = decompose.to_dict()
        plan_dict["rewrite"] = rewrite.to_dict()

        # 4) retrieve（对每个原子子问题分别检索后合并）
        hits, pipeline_route = retrieve_sub_queries(
            tools, plan, decompose.sub_queries, tools_used
        )

        q = plan.rewritten_query
        sub_query_texts = [sq.text for sq in decompose.sub_queries]

        # 5) judge evidence
        ev = judge_evidence(
            hits,
            pipeline_route=pipeline_route,
            plan_confidence=plan.confidence,
            query=q,
            route=plan.route,
        )
        tools_used.append("judge_evidence")

        # 5b) supplement retrieval
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
                hits = merge_hits(hits, extra)
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

        # 6) generate（整合各子问题检索结果）
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
                    + ("；".join(ev.missing_aspects) if ev.missing_aspects else "")
                )
            cites = ev.citations
        else:
            answer, cites = tools.generate_answer(
                q,
                hits,
                agent_route=plan.route,
                pipeline_route=pipeline_route,
                sub_queries=sub_query_texts if len(sub_query_texts) > 1 else None,
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
            plan=plan_dict,
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
            "sub_queries": (resp.plan or {}).get("sub_queries", []),
            "decomposer_type": ((resp.plan or {}).get("decompose") or {}).get(
                "decomposer_type"
            ),
            "rewriter_type": ((resp.plan or {}).get("rewrite") or {}).get(
                "rewriter_type"
            ),
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
