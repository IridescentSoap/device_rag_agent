"""text_utils 单元测试。"""

import unittest

from rag.text_utils import split_oversized


class TestTextUtils(unittest.TestCase):
    def test_split_oversized_short_unchanged(self):
        text = "短段落，不触发切分。"
        self.assertEqual(split_oversized(text, 1024, 128), [text])

    def test_split_oversized_long_prefers_sentence_boundaries(self):
        sentence = "系统会对雷达报文进行严格校验，格式错误时丢弃处理。"
        text = sentence * 60
        parts = split_oversized(text, chunk_size=400, chunk_overlap=40)
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLessEqual(len(p), 450)
            self.assertTrue(
                p.endswith(("。", "！", "？", ".", "!", "?")) or len(p) < 400,
                msg=f"chunk 末尾非句边界: ...{p[-30:]!r}",
            )


if __name__ == "__main__":
    unittest.main()
