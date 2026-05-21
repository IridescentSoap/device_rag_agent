"""结构化分块单元测试。"""

import unittest

from rag.structured_chunking import (
    chunk_manual_text_structured,
    parse_markdown_sections,
)


SAMPLE = """# 手册

## 目 录

1.1. 设计原则 ...... 1
1.2. 参考文献 ...... 2

## 1.1. 设计原则

这是设计原则正文，需要足够长度才能通过过滤。系统应满足可靠性要求。

## 2.1.4. QNH 数据处理

QNH 用于高度换算。当外部链路 QNH 跳变过大时不允许修改。
"""


class TestStructuredChunking(unittest.TestCase):
    def test_parse_sections(self):
        secs = parse_markdown_sections(SAMPLE)
        titles = [s.title for s in secs]
        self.assertIn("目 录", titles)
        self.assertIn("1.1. 设计原则", titles)
        self.assertIn("2.1.4. QNH 数据处理", titles)

    def test_toc_filtered(self):
        bundle = chunk_manual_text_structured(
            SAMPLE, "test_doc", chunk_size=200, filter_toc=True
        )
        texts = " ".join(c.text for c in bundle.children)
        self.assertNotIn("......", texts)
        self.assertIn("QNH", texts)
        self.assertTrue(any("设计原则" in c.text for c in bundle.children))

    def test_parent_child_link(self):
        bundle = chunk_manual_text_structured(
            SAMPLE, "test_doc", chunk_size=200, filter_toc=True
        )
        self.assertTrue(bundle.parents)
        child = next(c for c in bundle.children if "QNH" in c.text)
        pid = child.meta["parent_id"]
        self.assertIn(pid, {p.chunk_id for p in bundle.parents})
        parent = next(p for p in bundle.parents if p.chunk_id == pid)
        self.assertIn("QNH", parent.text)


if __name__ == "__main__":
    unittest.main()
