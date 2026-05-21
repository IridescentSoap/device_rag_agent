# device_rag_llm

空管设备运维 **Agentic RAG** 知识库系统：底层双源检索（手册 + 运维日志），上层 Agent 编排（查询规划、证据判断、补充检索、回答生成）。

---

## 功能概览

| 层级 | 能力 |
|------|------|
| **rag/** | BM25 + 向量（Qdrant）、RRF 融合、Rerank、上下文扩展、LLM 生成 |
| **agent/** | 多轮改写、规则/LLM Planner、分源检索、证据充分性判断、补充检索循环 |
| **api/** | FastAPI `/ask`、`/health`、`/metrics` |
| **scripts/** | 命令行 Demo `run_agent_demo.py` |

典型 Agent 流程：

```text
rewrite_query → plan_query → retrieve → judge_evidence
  → [supplement_search → judge_evidence]* → generate_answer
```

---

## 环境准备

```bash
# 建议使用 Python 3.10+（conda 示例）
conda create -n rag python=3.10 -y
conda activate rag

cd device_rag_llm
pip install -r requirements-rag.txt

cp .env.example .env
# 编辑 .env：LLM_API_KEY、QDRANT_URL、设备相关变量等
```

主要环境变量见 [`.env.example`](.env.example)。

---

## 建索引

```bash
# 仅运维日志
python build_index.py --logs data/filtered_maintenance_data.csv

# 手册 + 日志
python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv
```

向量写入 Qdrant 前需先启动 Qdrant 服务（见下文）。

---

## 底层 RAG 问答（CLI）

```bash
# 双源混合问答
python query_rag.py -q "你的问题"

# 仅日志检索（BM25 + 向量 → RRF → rerank 阈值过滤）
python query_log_rag.py -q "某机型某故障现象是否会影响业务" --threshold 0.7 --final-topk 10

# 阈值后候选不足时自动补齐
python query_log_rag.py -q "某机型某故障现象是否会影响业务" --threshold 0.7 --final-topk 10 --fill-shortage
```

---

## Agentic RAG

### 命令行 Demo

```bash
# 完整链路（需 LLM_API_KEY）
python scripts/run_agent_demo.py -q "某设备黑屏是否影响业务，应该如何处理？"

# 仅检索 + 证据判断，快速模式，JSON 输出
python scripts/run_agent_demo.py -q "某设备黑屏是否影响业务，应该如何处理？" \
  --retrieve-only --fast --json

# 启用 LLM Planner（失败自动回退规则 Planner）
python scripts/run_agent_demo.py -q "某设备黑屏是否影响业务，应该如何处理？" \
  --retrieve-only --fast --json --llm-planner

# 补充检索轮数（默认 1）
python scripts/run_agent_demo.py -q "你的问题" --max-supplement-rounds 1
```

响应 JSON 中关注字段：`plan`（含 `planner_type`、`route`、`sub_queries`）、`evidence`（含 `missing_aspects`、`supplement_queries`）、`supplement_rounds`、`tools_used`。

### API 服务

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

`POST /ask` 请求体示例：

```json
{
  "query": "某设备黑屏是否影响业务，应该如何处理？",
  "history": [],
  "skip_llm": false,
  "fast_mode": true,
  "max_supplement_rounds": 1,
  "use_llm_planner": true
}
```

### 查询规划（Planner）

1. **默认**：规则 Planner（关键词 + 设备名匹配），`plan.planner_type` 为 `rule`。
2. **启用 LLM Planner**：
   - 环境变量 `AGENT_LLM_PLANNER=1`，或
   - API `use_llm_planner: true`，或
   - Demo `--llm-planner`（显式开启；未加该参数时 demo 不会读 `AGENT_LLM_PLANNER`）。
3. **适用场景**：复杂问题、多轮追问、手册+日志综合排查、子问题拆解。
4. **模型**：`AGENT_LLM_PLANNER_MODEL=qwen-plus`（默认与 `RAG_LLM_MODEL` 一致）；`AGENT_LLM_PLANNER_TEMPERATURE=0`。
5. **兜底**：LLM 调用或 JSON 校验失败时回退规则 Planner（`planner_type` 为 `rule_fallback`）。

### 补充检索

证据不足（缺手册/日志、缺处置步骤等）时，按 `evidence.supplement_queries` 自动补检并合并 hits，轮数由 `max_supplement_rounds` 控制（默认 1）。

---

## 手册处理流水线（PDF → txt → 索引）

1. **MinerU 转 PDF**  
   将 PDF 转为 txt/md，产物例如：`data/manual_converted/mineru_out`（命令以本地 MinerU 版本为准）。

2. **整理 + 质检 + 建索引**

   ```bash
   python manual_pipeline.py --converted-dir data/manual_converted/mineru_out
   ```

3. **质检报告**  
   `data/manual_qc/report.json`；`suspicious=true` 的文件建议用 PaddleOCR 重跑后覆盖 converted 目录。

4. **仅整理与质检（不建索引）**

   ```bash
   python manual_pipeline.py --converted-dir data/manual_converted/mineru_out --skip-build-index
   ```

5. **手动建索引（可选）**

   ```bash
   python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv
   ```

6. **一体化（PDF → MinerU → manual_pipeline）**

   ```bash
   python run_mineru_pipeline.py \
     --pdf-dir data/manual_pdf \
     --mineru-cmd-template "mineru -i {input} -o {output}"
   ```

说明：

- MinerU 各版本 CLI 不一致，使用 `--mineru-cmd-template` 适配；模板须包含 `{input}` 与 `{output}`。
- 仅 PDF 解析、不跑后续：加 `--skip-manual-pipeline`。
- manual 入口为纯文本 `.txt`；`manual_pipeline` 会将 `.md`/`.txt` 统一为 `data/manual_txt/*.txt`。
- Qdrant 物理落盘目录由 Qdrant 启动配置决定，不由 `build_index.py --index-dir` 决定。

---

## Qdrant 本地启动（离线二进制）

若你本地已有 qdrant 安装包（例如 `/root/autodl-tmp/qdrant-x86_64-unknown-linux-musl.tar.gz`）：

1) 解压

```bash
mkdir -p /root/autodl-tmp/qdrant-bin
tar -xzf /root/autodl-tmp/qdrant-x86_64-unknown-linux-musl.tar.gz -C /root/autodl-tmp/qdrant-bin
```

2) 启动（复用本项目向量存储目录）

```bash
QDRANT__STORAGE__STORAGE_PATH="/root/autodl-tmp/device_rag_llm/data/qdrant_storage" \
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
"/root/autodl-tmp/qdrant-bin/qdrant"
```

3) 校验服务是否可用

```bash
curl http://localhost:6333/collections
```

4) 再运行日志检索

```bash
python query_log_rag.py -q "你的问题" --threshold 0.7 --final-topk 10 --fill-shortage
```

---

## 项目结构

```text
device_rag_llm/
├── rag/                 # 底层 RAG（检索、重排、生成）
├── agent/               # Agent 编排（planner、evidence、executor、workflow）
├── api/server.py        # FastAPI
├── scripts/             # Demo 与工具脚本
├── build_index.py       # 建索引 CLI
├── query_rag.py         # 双源问答 CLI
├── query_log_rag.py     # 日志检索 CLI
├── manual_pipeline.py   # 手册整理流水线
├── data/                # 索引、Qdrant 存储、样本数据
└── requirements-rag.txt
```

### rag/ 模块说明

| 模块 | 说明 |
|------|------|
| `config.py` | 索引目录、Embedding/LLM/Rerank 等环境变量 |
| `index_store.py` | 双源 CorpusIndex（手册 / 日志） |
| `retrieve.py` | Query 路由 + 分源 RRF |
| `rerank.py` | DashScope 或本地 Qwen3-Reranker |
| `pipeline.py` | 检索 → 重排 → 扩展 → 生成 |
| `llm.py` | OpenAI 兼容生成接口 |

### agent/ 模块说明

| 模块 | 说明 |
|------|------|
| `planner.py` | 规则 Planner + LLM Planner（JSON 校验与回退） |
| `evidence.py` | 证据判断与 `supplement_queries` 生成 |
| `executor.py` | 线性执行器（LangGraph 不可用时的回退） |
| `workflow.py` | LangGraph 工作流（优先） |
| `tools.py` | 封装 `search_manual` / `search_logs` / `hybrid_search` / `generate_answer` |

---

## 日志与监控

Agent 查询轨迹写入 `logs/query_log.jsonl`（含 `route`、`tools_used`、`supplement_rounds` 等）。API `GET /metrics` 可查看聚合指标。

---

## 说明

- 生成与 Planner 默认使用 DashScope 兼容接口（`qwen-plus`），可通过 `DASHSCOPE_BASE_URL` 对接本地 vLLM 等。
- 安装 LangGraph 后 `AgentWorkflow` 走图编排；未安装时自动回退 `AgentExecutor`，行为保持一致。
- 勿将 `.env` 提交到版本库；密钥仅放在本地 `.env`。
