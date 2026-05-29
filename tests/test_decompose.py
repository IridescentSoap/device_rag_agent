"""Query Decomposition 单元测试。"""

from __future__ import annotations

import unittest

from agent.decompose import (
    decompose_query,
    needs_decomposition,
    rule_decompose_query,
)


class TestDecompose(unittest.TestCase):
    def test_impact_and_handling_split(self) -> None:
        q = "某雷达黑屏是否影响业务，应该如何处理？"
        result = rule_decompose_query(
            q,
            route="hybrid_diagnosis",
            needs_manual=True,
            needs_log=True,
        )
        texts = [sq.text for sq in result.sub_queries]
        self.assertGreaterEqual(len(texts), 2)
        self.assertTrue(any("影响" in t for t in texts))
        self.assertTrue(any("处理" in t or "如何" in t for t in texts))

    def test_single_intent_no_split(self) -> None:
        q = "自动化系统 QNH 参数如何配置？"
        result = rule_decompose_query(
            q,
            route="manual_query",
            needs_manual=True,
            needs_log=False,
        )
        self.assertEqual(len(result.sub_queries), 1)
        self.assertIn("QNH", result.sub_queries[0].text)

    def test_hybrid_dual_source_variants(self) -> None:
        q = "雷达告警怎么处理"
        result = rule_decompose_query(
            q,
            route="hybrid_diagnosis",
            needs_manual=True,
            needs_log=True,
        )
        self.assertGreaterEqual(len(result.sub_queries), 3)
        aspects = {sq.aspect for sq in result.sub_queries}
        self.assertIn("manual", aspects)
        self.assertIn("log", aspects)

    def test_needs_decomposition(self) -> None:
        self.assertTrue(
            needs_decomposition(
                "黑屏是否影响业务，应该如何处理？",
                "hybrid_diagnosis",
            )
        )
        self.assertFalse(
            needs_decomposition(
                "QNH 参数说明",
                "manual_query",
            )
        )

    def test_insufficient_route_empty(self) -> None:
        result = decompose_query(
            "怎么办",
            route="insufficient_evidence",
        )
        self.assertEqual(result.sub_queries, [])
        self.assertEqual(result.decomposer_type, "none")

    def test_cause_then_handling(self) -> None:
        q = "进程单故障的原因，如何恢复？"
        result = rule_decompose_query(
            q,
            route="hybrid_diagnosis",
            needs_manual=True,
            needs_log=True,
        )
        texts = [sq.text for sq in result.sub_queries]
        self.assertGreaterEqual(len(texts), 2)


if __name__ == "__main__":
    unittest.main()
