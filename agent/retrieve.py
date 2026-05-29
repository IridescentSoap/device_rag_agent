"""按原子子问题分路检索并合并 hits。"""

from __future__ import annotations

from agent.decompose import SubQuery
from agent.hits import merge_hits
from agent.state import PlanResult
from agent.tools import RagTools
from rag.rerank import RerankHit


def _search_one(
    tools: RagTools,
    plan: PlanResult,
    sq: SubQuery,
) -> tuple[list[RerankHit], str]:
    if sq.prefer_manual and not sq.prefer_log:
        return tools.search_manual(sq.text), "manual_heavy"
    if sq.prefer_log and not sq.prefer_manual:
        return tools.search_logs(sq.text), "log_heavy"

    if plan.route == "manual_query":
        return tools.search_manual(sq.text), "manual_heavy"
    if plan.route == "log_case_query":
        return tools.search_logs(sq.text), "log_heavy"

    hits, route = tools.hybrid_search(sq.text)
    return hits, route


def _resolve_pipeline_route(routes: list[str]) -> str:
    if not routes:
        return "balanced"
    if all(r == "manual_heavy" for r in routes):
        return "manual_heavy"
    if all(r == "log_heavy" for r in routes):
        return "log_heavy"
    return "balanced"


def retrieve_sub_queries(
    tools: RagTools,
    plan: PlanResult,
    sub_queries: list[SubQuery],
    tools_used: list[str],
) -> tuple[list[RerankHit], str]:
    """
    对每个原子子问题分别检索，合并去重后返回 hits 与 pipeline_route。
    """
    if not sub_queries:
        sub_queries = [SubQuery(text=plan.rewritten_query)]

    all_batches: list[list[RerankHit]] = []
    routes: list[str] = []

    for sq in sub_queries:
        batch, route = _search_one(tools, plan, sq)
        all_batches.append(batch)
        routes.append(route)
        if sq.prefer_manual and not sq.prefer_log:
            tools_used.append("search_manual")
        elif sq.prefer_log and not sq.prefer_manual:
            tools_used.append("search_logs")
        elif plan.route == "manual_query":
            tools_used.append("search_manual")
        elif plan.route == "log_case_query":
            tools_used.append("search_logs")
        else:
            tools_used.append("hybrid_search")

    if len(sub_queries) > 1:
        tools_used.append("decompose_retrieve")

    merged = merge_hits(*all_batches) if all_batches else []
    pipeline_route = _resolve_pipeline_route(routes)
    return merged, pipeline_route
