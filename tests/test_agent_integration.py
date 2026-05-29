"""Agent 流水线集成测试（Mock 检索与 LLM，无需索引/API Key）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.executor import AgentExecutor
from agent.tools import RagTools
from rag.rerank import RerankHit
from rag.schemas import ChunkRecord


def _hit(
    chunk_id: str,
    *,
    source: str = "log",
    score: float = 0.9,
    text: str = "参考资料片段",
) -> RerankHit:
    return RerankHit(
        chunk=ChunkRecord(chunk_id=chunk_id, source=source, text=text),  # type: ignore[arg-type]
        score=score,
    )


class FakeRagTools(RagTools):
    """绕过索引加载，记录检索调用。"""

    def __init__(self) -> None:
        self.fast_mode = False
        self.index_dir = None
        self.device = None
        self.hybrid_calls: list[str] = []
        self.manual_calls: list[str] = []
        self.log_calls: list[str] = []

    @property
    def pipe(self):  # type: ignore[override]
        raise RuntimeError("integration test should not load pipeline")

    def hybrid_search(self, query: str, **kwargs) -> tuple[list[RerankHit], str]:
        self.hybrid_calls.append(query)
        cid = f"log-{len(self.hybrid_calls)}"
        return [_hit(cid, source="log", text=f"日志:{query}")], "balanced"

    def search_manual(self, query: str, **kwargs) -> list[RerankHit]:
        self.manual_calls.append(query)
        cid = f"manual-{len(self.manual_calls)}"
        return [_hit(cid, source="manual", text=f"手册:{query}")]

    def search_logs(self, query: str, **kwargs) -> list[RerankHit]:
        self.log_calls.append(query)
        cid = f"logonly-{len(self.log_calls)}"
        return [_hit(cid, source="log", text=f"案例:{query}")]

    def generate_answer(self, query, hits, **kwargs) -> tuple[str, list[str]]:
        sub = kwargs.get("sub_queries") or []
        cites = [h.chunk.chunk_id for h in hits[:3]]
        if sub:
            return f"整合回答（{len(sub)} 个子问题）: {query}", cites
        return f"回答: {query}", cites

    def mode_config(self) -> dict:
        return {"enabled": False}


class TestAgentIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = FakeRagTools()
        self.executor = AgentExecutor(tools=self.tools)

    def test_compound_query_decompose_multi_retrieve(self) -> None:
        resp = self.executor.run(
            "某雷达黑屏是否影响业务，应该如何处理？",
            skip_llm=True,
            fast_mode=False,
        )
        self.assertGreaterEqual(len(resp.plan.get("sub_queries") or []), 2)
        self.assertIn("decompose_retrieve", resp.tools_used)
        self.assertGreaterEqual(len(self.tools.hybrid_calls), 2)
        self.assertIn("decompose_query", resp.tools_used)
        self.assertGreater(len(resp.citations), 0)

    def test_rule_followup_rewrite_pipeline(self) -> None:
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        resp = self.executor.run(
            "需要重启吗？",
            history=history,
            skip_llm=True,
            fast_mode=False,
        )
        rewrite = (resp.plan or {}).get("rewrite") or {}
        self.assertEqual(rewrite.get("rewriter_type"), "rule")
        self.assertIn("E001", resp.rewritten_query)
        self.assertIn("rewrite_query", resp.tools_used)

    @patch("rag.llm.chat")
    def test_llm_followup_rewrite_pipeline(self, mock_chat) -> None:
        mock_chat.return_value = "雷达黑屏 E001 故障是否需要重启"
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        resp = self.executor.run(
            "需要重启吗？",
            history=history,
            skip_llm=True,
            use_llm_rewrite=True,
            fast_mode=False,
        )
        rewrite = (resp.plan or {}).get("rewrite") or {}
        self.assertEqual(rewrite.get("rewriter_type"), "llm")
        self.assertIn("rewrite_query_llm", resp.tools_used)
        self.assertNotIn("补充追问", resp.rewritten_query)
        self.assertIn("重启", resp.rewritten_query)
        mock_chat.assert_called()

    @patch("rag.llm.chat")
    def test_llm_rewrite_fallback_then_decompose(self, mock_chat) -> None:
        mock_chat.side_effect = RuntimeError("llm unavailable")
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        resp = self.executor.run(
            "需要重启吗？",
            history=history,
            skip_llm=True,
            use_llm_rewrite=True,
            fast_mode=False,
        )
        rewrite = (resp.plan or {}).get("rewrite") or {}
        self.assertEqual(rewrite.get("rewriter_type"), "rule_fallback")
        self.assertIn("rewrite_query_rule_fallback", resp.tools_used)
        self.assertIn("decompose_query", resp.tools_used)

    def test_manual_query_single_retrieve(self) -> None:
        resp = self.executor.run(
            "自动化系统 QNH 参数如何配置？",
            skip_llm=True,
            fast_mode=False,
        )
        self.assertEqual(resp.route, "manual_query")
        self.assertEqual(len(resp.plan.get("sub_queries") or []), 1)
        self.assertEqual(len(self.tools.manual_calls), 1)
        self.assertEqual(len(self.tools.hybrid_calls), 0)

    def test_generate_with_sub_queries(self) -> None:
        resp = self.executor.run(
            "某雷达黑屏是否影响业务，应该如何处理？",
            skip_llm=False,
            fast_mode=False,
        )
        self.assertIn("generate_answer", resp.tools_used)
        self.assertIn("整合回答", resp.answer)
        self.assertGreaterEqual(len(resp.plan.get("sub_queries") or []), 2)


if __name__ == "__main__":
    unittest.main()
