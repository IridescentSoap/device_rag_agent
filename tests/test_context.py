"""rewrite_query 单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.context import (
    build_llm_rewrite_prompt,
    format_history_for_llm_rewrite,
    llm_rewrite_query,
    needs_history,
    rewrite_query,
    rewrite_query_with_meta,
    rule_rewrite_query,
    should_use_llm_rewrite,
)


class TestContextRewrite(unittest.TestCase):
    def test_needs_history_followup(self) -> None:
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        self.assertTrue(needs_history("需要重启吗？", history))
        self.assertFalse(needs_history("雷达黑屏报错 E001 处置流程是什么", history))

    def test_rule_rewrite_concat(self) -> None:
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        result = rule_rewrite_query("需要重启吗？", history)
        self.assertEqual(result.rewriter_type, "rule")
        self.assertIn("E001", result.query)
        self.assertIn("需要重启吗", result.query)

    def test_rule_rewrite_only_last_user_turn(self) -> None:
        history = [
            {"role": "user", "content": "自动化系统 QNH 参数范围是多少"},
            {"role": "assistant", "content": "手册记载为 860-1100"},
            {"role": "user", "content": "雷达黑屏报错 E001"},
        ]
        result = rule_rewrite_query("需要重启吗？", history)
        self.assertIn("E001", result.query)
        self.assertNotIn("QNH", result.query)

    def test_llm_prompt_includes_extended_history(self) -> None:
        history = [
            {"role": "user", "content": "雷达A黑屏报错 E001"},
            {"role": "assistant", "content": "可能是电源模块故障"},
            {"role": "user", "content": "已更换电源模块"},
            {"role": "assistant", "content": "若仍黑屏建议查背板"},
            {"role": "user", "content": "背板指示灯正常"},
        ]
        prompt = build_llm_rewrite_prompt("那它还会复发吗？", history, history_turns=5)
        self.assertIn("雷达A黑屏报错 E001", prompt)
        self.assertIn("已更换电源模块", prompt)
        self.assertIn("最近 5 轮对话历史", prompt)

    def test_llm_history_window_excludes_older_turns(self) -> None:
        history = [
            {"role": "user", "content": "很早的雷达A黑屏 E001"},
            {"role": "assistant", "content": "回复1"},
            {"role": "user", "content": "中间追问"},
            {"role": "assistant", "content": "回复2"},
            {"role": "user", "content": "更近的追问"},
            {"role": "assistant", "content": "回复3"},
        ]
        prompt = build_llm_rewrite_prompt("那它还会复发吗？", history, history_turns=5)
        self.assertNotIn("很早的雷达A", prompt)
        self.assertIn("更近的追问", prompt)

    def test_format_history_for_llm_rewrite_window(self) -> None:
        history = [{"role": "user", "content": f"turn-{i}"} for i in range(8)]
        text = format_history_for_llm_rewrite(history, max_turns=5)
        self.assertIn("turn-3", text)
        self.assertIn("turn-7", text)
        self.assertNotIn("turn-2", text)

    def test_no_history_returns_original(self) -> None:
        result = rewrite_query_with_meta("自动化 QNH 如何配置？", history=[])
        self.assertEqual(result.query, "自动化 QNH 如何配置？")
        self.assertEqual(result.rewriter_type, "none")

    @patch("rag.llm.chat")
    def test_llm_rewrite(self, mock_chat) -> None:
        mock_chat.return_value = "雷达黑屏 E001 故障是否需要重启"
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        result = llm_rewrite_query("需要重启吗？", history)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.rewriter_type, "llm")
        self.assertIn("重启", result.query)
        self.assertNotIn("补充追问", result.query)

    @patch("rag.llm.chat")
    def test_rewrite_query_with_meta_llm(self, mock_chat) -> None:
        mock_chat.return_value = "雷达黑屏 E001 是否需要重启"
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        result = rewrite_query_with_meta(
            "需要重启吗？",
            history,
            use_llm_rewrite=True,
        )
        self.assertEqual(result.rewriter_type, "llm")
        self.assertIn("重启", result.query)

    @patch("rag.llm.chat")
    def test_llm_failure_fallback_rule(self, mock_chat) -> None:
        mock_chat.side_effect = RuntimeError("no api key")
        history = [{"role": "user", "content": "雷达黑屏报错 E001"}]
        result = rewrite_query_with_meta(
            "需要重启吗？",
            history,
            use_llm_rewrite=True,
        )
        self.assertEqual(result.rewriter_type, "rule_fallback")
        self.assertIn("补充追问", result.query)

    def test_should_use_llm_rewrite_explicit(self) -> None:
        history = [{"role": "user", "content": "x"}]
        self.assertTrue(
            should_use_llm_rewrite("需要重启吗", history, use_llm_rewrite=True)
        )
        self.assertFalse(
            should_use_llm_rewrite("独立完整问题", history=[], use_llm_rewrite=True)
        )

    def test_rewrite_query_compat(self) -> None:
        q = rewrite_query("完整问题", history=[], use_llm=True)
        self.assertEqual(q, "完整问题")


if __name__ == "__main__":
    unittest.main()
