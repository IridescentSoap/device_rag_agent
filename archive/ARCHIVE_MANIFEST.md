# 归档说明

以下文件为临时调试、一次性实验或重复入口，已从项目根目录移入 `archive/`，便于面试展示时保持目录清晰。

| 文件 | 理由 |
|------|------|
| `data_analyse.py` | 早期 CSV 探索脚本，含硬编码路径，非 RAG 主链路 |
| `docling_test.py` | Docling 解析实验，未接入建索引流水线 |
| `debug_bm25_log.py` | BM25 调试脚本 |
| `filter.py` | 日志 CSV 一次性过滤，产物已落盘 |
| `suggest_reference_ids.py` | 标注 gold 辅助，已由 `scripts/remap_eval_gold.py` 替代 |
| `eval_retrieval_quick.py` | 检索 Recall@K 冒烟评测，正式评测用 `eval_business_rag.py` |
| `query_rag.py` | 与 `query_dual_rag.py` 重复的双源 CLI；根目录保留同名 shim 转发 |

## 根目录保留的核心入口

| 类型 | 文件 |
|------|------|
| 建索引 | `build_index.py` |
| 双源 RAG | `query_dual_rag.py`（`query_rag.py` 为兼容 shim） |
| 分源 RAG | `query_log_rag.py`、`query_manual_rag.py` |
| 评测 | `eval_business_rag.py` |
| 手册流水线 | `manual_pipeline.py`、`mineru_to_pdf.py` |
| Agent | `scripts/run_agent_demo.py`、`api/server.py` |

## 归档脚本用法（需在项目根目录执行）

```bash
python archive/debug_bm25_log.py -q "查询" --topk 10
python archive/eval_retrieval_quick.py --sample 200
```
