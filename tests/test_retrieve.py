"""SubQuery / PlanResult 检索与聚合单元测试。"""

from __future__ import annotations

import unittest

from agent.decompose import SubQuery
from agent.retrieve import retrieve_for_plan, retrieve_sub_queries
from agent.state import PlanResult
from agent.tools import RagTools
from rag.rerank import RerankHit
from rag.schemas import ChunkRecord


def _hit(
    chunk_id: str,
    *,
    source: str = "log",
    score: float = 0.9,
    text: str = "片段",
) -> RerankHit:
    return RerankHit(
        chunk=ChunkRecord(chunk_id=chunk_id, source=source, text=text),  # type: ignore[arg-type]
        score=score,
    )


class FakeRagTools(RagTools):
    def __init__(self) -> None:
        self.fast_mode = False
        self.index_dir = None
        self.device = None
        self.hybrid_calls: list[str] = []
        self.manual_calls: list[str] = []
        self.log_calls: list[str] = []

    @property
    def pipe(self):  # type: ignore[override]
        raise RuntimeError("test should not load pipeline")

    def hybrid_search(self, query: str, **kwargs) -> tuple[list[RerankHit], str]:
        self.hybrid_calls.append(query)
        idx = len(self.hybrid_calls)
        return [_hit(f"hybrid-{idx}", score=0.5 + idx * 0.01)], "balanced"

    def search_manual(self, query: str, **kwargs) -> list[RerankHit]:
        self.manual_calls.append(query)
        idx = len(self.manual_calls)
        return [_hit(f"manual-{idx}", source="manual", score=0.6 + idx * 0.01)]

    def search_logs(self, query: str, **kwargs) -> list[RerankHit]:
        self.log_calls.append(query)
        idx = len(self.log_calls)
        return [_hit(f"log-{idx}", score=0.7 + idx * 0.01)]


def _plan(
    route: str,
    *,
    rewritten: str = "主问题",
    sub_queries: list[str] | None = None,
) -> PlanResult:
    needs_manual = route in ("manual_query", "hybrid_diagnosis", "follow_up_query")
    needs_log = route in ("log_case_query", "hybrid_diagnosis", "follow_up_query")
    return PlanResult(
        route=route,  # type: ignore[arg-type]
        rewritten_query=rewritten,
        sub_queries=list(sub_queries or []),
        needs_manual=needs_manual,
        needs_log=needs_log,
    )


class TestRetrieveForPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = FakeRagTools()

    def test_empty_sub_queries_falls_back_to_rewritten_query(self) -> None:
        plan = _plan("hybrid_diagnosis", rewritten="雷达黑屏怎么办")
        tools_used: list[str] = []
        hits, route = retrieve_for_plan(self.tools, plan, tools_used)

        self.assertEqual(self.tools.hybrid_calls, ["雷达黑屏怎么办"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(route, "balanced")
        self.assertIn("hybrid_search", tools_used)
        self.assertNotIn("decompose_retrieve", tools_used)

    def test_manual_query_multiple_sub_queries(self) -> None:
        plan = _plan("manual_query", rewritten="主问题")
        sub_queries = [
            SubQuery(text="QNH 参数"),
            SubQuery(text="QNH 配置步骤"),
        ]
        tools_used: list[str] = []
        hits, route = retrieve_sub_queries(self.tools, plan, sub_queries, tools_used)

        self.assertEqual(self.tools.manual_calls, ["QNH 参数", "QNH 配置步骤"])
        self.assertEqual(len(hits), 2)
        self.assertEqual(route, "manual_heavy")
        self.assertEqual(tools_used.count("search_manual_subquery"), 2)
        self.assertIn("decompose_retrieve", tools_used)

    def test_log_case_query_multiple_sub_queries(self) -> None:
        plan = _plan("log_case_query", rewritten="主问题")
        sub_queries = [
            SubQuery(text="雷达黑屏影响"),
            SubQuery(text="雷达黑屏案例"),
        ]
        tools_used: list[str] = []
        hits, route = retrieve_sub_queries(self.tools, plan, sub_queries, tools_used)

        self.assertEqual(self.tools.log_calls, ["雷达黑屏影响", "雷达黑屏案例"])
        self.assertEqual(len(hits), 2)
        self.assertEqual(route, "log_heavy")
        self.assertEqual(tools_used.count("search_logs_subquery"), 2)

    def test_hybrid_diagnosis_multiple_sub_queries(self) -> None:
        plan = _plan("hybrid_diagnosis", rewritten="主问题")
        sub_queries = [
            SubQuery(text="是否影响业务"),
            SubQuery(text="如何处理"),
        ]
        tools_used: list[str] = []
        hits, route = retrieve_sub_queries(self.tools, plan, sub_queries, tools_used)

        self.assertEqual(self.tools.hybrid_calls, ["是否影响业务", "如何处理"])
        self.assertEqual(len(hits), 2)
        self.assertEqual(route, "balanced")
        self.assertEqual(tools_used.count("hybrid_search_subquery"), 2)

    def test_subquery_prefer_manual_overrides_route(self) -> None:
        plan = _plan("hybrid_diagnosis", rewritten="主问题")
        sub_queries = [SubQuery(text="QNH 参数", prefer_manual=True)]
        tools_used: list[str] = []
        hits, route = retrieve_sub_queries(self.tools, plan, sub_queries, tools_used)

        self.assertEqual(self.tools.manual_calls, ["QNH 参数"])
        self.assertEqual(len(self.tools.hybrid_calls), 0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(route, "manual_heavy")

    def test_empty_sub_queries_falls_back_in_retrieve_sub_queries(self) -> None:
        plan = _plan("hybrid_diagnosis", rewritten="雷达黑屏怎么办")
        hits, route = retrieve_sub_queries(self.tools, plan, [], [])

        self.assertEqual(self.tools.hybrid_calls, ["雷达黑屏怎么办"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(route, "balanced")

    def test_merge_keeps_highest_score_for_same_chunk(self) -> None:
        shared = _hit("dup-1", score=0.5)
        high = _hit("dup-1", score=0.95)

        class DupTools(FakeRagTools):
            def hybrid_search(self, query: str, **kwargs) -> tuple[list[RerankHit], str]:
                self.hybrid_calls.append(query)
                if len(self.hybrid_calls) == 1:
                    return [shared], "balanced"
                return [high], "balanced"

        tools = DupTools()
        plan = _plan("hybrid_diagnosis", rewritten="主问题")
        sub_queries = [SubQuery(text="q1"), SubQuery(text="q2")]
        hits, _ = retrieve_sub_queries(tools, plan, sub_queries, [])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.chunk_id, "dup-1")
        self.assertAlmostEqual(float(hits[0].score), 0.95)


if __name__ == "__main__":
    unittest.main()
