"""多手册目录路由单元测试。"""

import unittest

from rag.manual_catalog import ManualDocProfile, ManualDocCatalog, build_manual_catalog
from rag.schemas import ChunkRecord


class TestManualCatalog(unittest.TestCase):
    def test_single_doc_no_filter_needed(self):
        chunks = [
            ChunkRecord(
                "A#00000",
                "manual",
                "QNH 数据处理",
                {"doc_id": "手册A", "chapter_path": "监视 > QNH"},
            )
        ]
        cat = build_manual_catalog(chunks)
        self.assertEqual(cat.select_doc_ids("QNH 作用"), ["手册A"])

    def test_two_docs_routing(self):
        chunks = [
            ChunkRecord(
                "自动化#00000",
                "manual",
                "进港排序 AMAN 推演",
                {"doc_id": "自动化系统技术手册", "chapter_path": "进港排序"},
            ),
            ChunkRecord(
                "接口#00000",
                "manual",
                "OLDI 移交 AIDC 接口",
                {"doc_id": "接口说明手册", "chapter_path": "OLDI 移交"},
            ),
        ]
        cat = build_manual_catalog(chunks)
        picked = cat.select_doc_ids("OLDI 移交如何描述", top_k=1)
        self.assertEqual(picked, ["接口说明手册"])

    def test_resolve_returns_none_when_one_doc(self):
        from rag.index_store import CorpusIndex
        from rag.retrieve import resolve_manual_doc_ids

        corp = CorpusIndex("manual")
        corp.chunks = [
            ChunkRecord("x#0", "manual", "t", {"doc_id": "only"}),
        ]
        corp.doc_catalog = build_manual_catalog(corp.chunks)
        self.assertIsNone(resolve_manual_doc_ids(corp, "任意问题"))


if __name__ == "__main__":
    unittest.main()
