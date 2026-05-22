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

    def test_outline_chapter_path(self):
        text = """# 封面

## 1. 前言

## 1.1. 设计原则

前言子节正文。

## 2. 功能描述

## 2.1. 监视数据处理

## 2.1.1. 雷达前置处理

雷达正文。
"""
        secs = {s.title: s for s in parse_markdown_sections(text)}
        self.assertEqual(
            secs["1.1. 设计原则"].chapter_path,
            "1. 前言 > 1.1. 设计原则",
        )
        self.assertEqual(
            secs["2.1.1. 雷达前置处理"].chapter_path,
            "2. 功能描述 > 2.1. 监视数据处理 > 2.1.1. 雷达前置处理",
        )
        self.assertEqual(secs["2.1.1. 雷达前置处理"].heading_level, 3)

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

    def test_inline_list_not_reset_stack(self):
        text = """## 2. 功能描述

## 2.1. 监视数据处理

## 2.1.1. 雷达前置处理

## 2.1.1.2. 坐标变换

正文。

## 1. 单雷达直角坐标转换为经纬度

步骤一正文。

## 2. 经纬度转换为相对中心点的直角坐标 XY

步骤二正文。

## 2.1.1.3. 异常数据处理

后续正文。
"""
        secs = {s.title: s for s in parse_markdown_sections(text)}
        self.assertIn("2.1.1.2. 坐标变换", secs["1. 单雷达直角坐标转换为经纬度"].chapter_path)
        self.assertIn("2.1.1.3. 异常数据处理", secs["2.1.1.3. 异常数据处理"].chapter_path)
        self.assertNotIn("图 ", secs["2.1.1.3. 异常数据处理"].chapter_path)

    def test_decorative_heading_merged_into_body(self):
        text = """## 2. 功能描述

## 2.1. 监视数据处理

## 2.1.1. 雷达前置处理

## 2.1.1.2. 坐标变换

说明文字。

## 图 经纬度转换为 XY

## 2.1.1.3. 异常数据处理

异常正文。
"""
        secs = {s.title: s for s in parse_markdown_sections(text)}
        self.assertNotIn("图 经纬度转换为 XY", secs)
        self.assertIn("图 经纬度转换为 XY", secs["2.1.1.2. 坐标变换"].body)
        self.assertEqual(
            secs["2.1.1.3. 异常数据处理"].chapter_path,
            "2. 功能描述 > 2.1. 监视数据处理 > 2.1.1. 雷达前置处理 > 2.1.1.3. 异常数据处理",
        )

    def test_new_top_chapter_after_deep_stack(self):
        text = """## 2. 功能描述

## 2.16. 其他功能

## 2.16.1. 模拟发报工具

正文。

## 3. 外部交互

## 3.1. 主备同步

交互正文。
"""
        secs = {s.title: s for s in parse_markdown_sections(text)}
        self.assertEqual(secs["3. 外部交互"].heading_level, 1)
        self.assertEqual(secs["3. 外部交互"].chapter_path, "3. 外部交互")
        self.assertEqual(
            secs["3.1. 主备同步"].chapter_path,
            "3. 外部交互 > 3.1. 主备同步",
        )


if __name__ == "__main__":
    unittest.main()
