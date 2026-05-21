# 结构化 RAG 实现流程说明（v2）

本文说明新一版算法在仓库中的落地方式与端到端数据流。

## 1. 总览

| 阶段 | 模块 | 作用 |
|------|------|------|
| 离线建库 | `rag/structured_chunking.py` | 章节分块、目录过滤、父子块 |
| 离线建库 | `build_index.py` | 子块入 Qdrant/BM25，父块写 `parents.jsonl` |
| 在线检索 | `rag/retrieve.py` | 双源 BM25+向量+RRF+路由+分源 Rerank（未改） |
| 生成前扩展 | `rag/context_expand.py` | 父块全文 / 同节邻块 / 日志原因处置 |

## 2. 索引构建

### 2.1 章节解析

`parse_markdown_sections()` 扫描 `^#{1,6}\s+标题`，维护标题栈，得到若干 `Section`（`chapter_path`、`body`）。

### 2.2 目录过滤

`is_toc_or_noise_section()` 过滤：

- 标题含「目录」「修订记录」等
- 正文大量 `1.1. xxx ...... 页码` 行
- 文首修订记录 `<table>` 块（`_preamble`）

被过滤的节 **不生成 child/parent**，不进检索索引。

### 2.3 父子块

对每个保留节：

- **parent**：`{doc_id}#sec_{序号}`，存整节正文 → `data/rag_index/manual/parents.jsonl`
- **child**：节内段落+滑动窗口切分，`chunk_id` 仍为 `{doc_id}#{全局序号:05d}`，meta 含 `parent_id`、`chunk_index_in_section`

仅 **child** 调用 `CorpusIndex.build()` 写入 Qdrant 与 BM25。

### 2.4 建索引命令

```bash
# 默认：结构化分块
python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv --recreate

# 回退旧分块
python build_index.py --manual-dir data/manual_txt --legacy-chunking --recreate
```

重建后需更新评测 gold（见 `scripts/remap_eval_gold.py`）。

## 3. 检索（与 P0.1 相同）

`DualSourceRagPipeline.retrieve()`：分源召回 → 路由 → 分源 Rerank → 配额合并。检索对象始终是 **child chunk_id**。

## 4. 生成前上下文扩展

`expand_context()` → `manual_body_for_prompt()`：

1. 若 `RAG_MANUAL_EXPAND_USE_PARENT=1` 且存在 `parent_id`：输出「命中子块摘录 + 所属章节全文」
2. 否则：子块正文 + **同 `parent_id` 内** `chunk_index_in_section±1` 邻块
3. 日志：`log_chunk_body_for_prompt()` 拼接 `meta.cause` / `meta.solution`

环境变量见 `rag/config.py` 中 `RAG_MANUAL_*`。

## 5. 问答入口

```bash
python query_dual_rag.py -q "QNH 数据处理的作用是什么？"
```

## 6. 多手册：文档级路由（已实现骨架）

当 `manual` 索引内 **手册数 ≥ 2** 且 `RAG_MANUAL_DOC_ROUTING=1` 时：

1. 建库生成 `data/rag_index/manual/docs.jsonl`（每本手册的标题、章节名、路由用文本）
2. 检索前 `ManualDocCatalog.select_doc_ids(query)` → Top-K 本手册（默认 2）
3. BM25 / 向量仅在对应 `doc_id` 的 child 块上召回（Qdrant payload 含顶层 `doc_id`）

**仅 1 本手册时自动跳过**，与现网行为一致。

环境变量：`RAG_MANUAL_DOC_TOPK`、`RAG_MANUAL_DOC_ROUTING_MIN_DOCS`。

## 7. 评测 gold 重映射

```bash
cp data/rag_index/manual/chunks.jsonl data/rag_index/manual/chunks.jsonl.bak
python build_index.py ... --recreate
python scripts/remap_eval_gold.py \
  --old-chunks data/rag_index/manual/chunks.jsonl.bak \
  --new-chunks data/rag_index/manual/chunks.jsonl \
  --eval data/eval/business_eval_30.jsonl
python eval_business_rag.py
```
