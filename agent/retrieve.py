"""按 PlanResult / SubQuery 分路检索并合并 hits。"""

from __future__ import annotations

from collections import Counter

from agent.decompose import SubQuery
from agent.hits import merge_hits
from agent.state import PlanResult
from agent.tools import RagTools
from rag.rerank import RerankHit


def _tool_name_for_route(plan: PlanResult, *, multi: bool) -> str:
    suffix = "_subquery" if multi else ""
    if plan.route == "manual_query":
        return f"search_manual{suffix}"
    if plan.route == "log_case_query":
        return f"search_logs{suffix}"
    return f"hybrid_search{suffix}"


def _search_one(
    tools: RagTools,
    plan: PlanResult,
    query_text: str,
    *,
    sq: SubQuery | None = None,
) -> tuple[list[RerankHit], str]:
    if sq is not None:
        if sq.prefer_manual and not sq.prefer_log:
            return tools.search_manual(query_text), "manual_heavy"
        if sq.prefer_log and not sq.prefer_manual:
            return tools.search_logs(query_text), "log_heavy"

    if plan.route == "manual_query":
        return tools.search_manual(query_text), "manual_heavy"
    if plan.route == "log_case_query":
        return tools.search_logs(query_text), "log_heavy"
    hits, route = tools.hybrid_search(query_text)
    return hits, route


def _resolve_pipeline_route(routes: list[str]) -> str:
    if not routes:
        return "balanced"
    if all(r == "manual_heavy" for r in routes):
        return "manual_heavy"
    if all(r == "log_heavy" for r in routes):
        return "log_heavy"
    most, freq = Counter(routes).most_common(1)[0]
    if freq > len(routes) / 2:
        return most
    return "balanced"


def retrieve_sub_queries(
    tools: RagTools,
    plan: PlanResult,
    sub_queries: list[SubQuery],
    tools_used: list[str],
) -> tuple[list[RerankHit], str]:
    """
    对每个原子子问题分别检索，合并去重后返回 hits 与 pipeline_route。

    sub_queries 为空时回退为 [plan.rewritten_query]。
    """
    if not sub_queries:
        sub_queries = [SubQuery(text=plan.rewritten_query)]

    multi = len(sub_queries) > 1
    all_batches: list[list[RerankHit]] = []
    routes: list[str] = []

    for sq in sub_queries:
        batch, route = _search_one(tools, plan, sq.text, sq=sq)
        all_batches.append(batch)
        routes.append(route)
        if sq.prefer_manual and not sq.prefer_log:
            tools_used.append("search_manual" + ("_subquery" if multi else ""))
        elif sq.prefer_log and not sq.prefer_manual:
            tools_used.append("search_logs" + ("_subquery" if multi else ""))
        elif plan.route == "manual_query":
            tools_used.append("search_manual" + ("_subquery" if multi else ""))
        elif plan.route == "log_case_query":
            tools_used.append("search_logs" + ("_subquery" if multi else ""))
        else:
            tools_used.append("hybrid_search" + ("_subquery" if multi else ""))

    if multi:
        tools_used.append("decompose_retrieve")

    merged = merge_hits(*all_batches) if all_batches else []
    pipeline_route = _resolve_pipeline_route(routes)
    return merged, pipeline_route


def retrieve_for_plan(
    tools: RagTools,
    plan: PlanResult,
    tools_used: list[str],
    *,
    queries: list[str] | None = None,
) -> tuple[list[RerankHit], str]:
    """
  补充检索等场景：按字符串 query 列表检索（无 SubQuery 偏好）。

  主检索请使用 retrieve_sub_queries + decompose 产出。
    """
    query_list = queries if queries is not None else plan.sub_queries
    if not query_list:
        query_list = [plan.rewritten_query]

    sub_queries = [SubQuery(text=q) for q in query_list]
    return retrieve_sub_queries(tools, plan, sub_queries, tools_used)
