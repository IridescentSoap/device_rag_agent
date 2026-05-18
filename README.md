rag/config.py	索引目录、embedding/LLM/rerank 等环境变量默认值
rag/schemas.py	ChunkRecord 数据结构
rag/chunking.py	手册纯文本分块、页眉噪声过滤
rag/embedder.py	bge-large-zh-v1.5（Sentence-Transformers）
rag/index_store.py	双源 CorpusIndex（手册 / 日志），BM25 + 向量，落盘 jsonl + npy
rag/ingest_logs.py	从 CSV 拼 [system][phenomenon][impact]，兼容 falut_* / fault_*
rag/retrieve.py	轻量 Query 路由 + 分源 RRF 融合
rag/rerank.py	DashScope TextReRank（默认 gte-rerank-v2，可用环境变量改为 qwen3-rerank 等）
rag/llm.py	OpenAI 兼容接口调用生成模型（默认 qwen-plus，与方案中本地 Qwen2.5-7B 可通过换 base_url/模型名对接）
rag/prompts.py	带引用与不确定性说明的系统提示
rag/pipeline.py	RagPipeline：检索 → 重排 → 相邻块扩展 → 生成
build_index.py	CLI：建索引
query_rag.py	CLI：问答
manual_pipeline.py	手册流水线：规范化文本 + 质检 + 可选触发建索引
run_mineru_pipeline.py	批量调用 MinerU 解析 PDF，并串联 manual_pipeline
requirements-rag.txt	依赖列表


pip install -r requirements-rag.txt
set DASHSCOPE_API_KEY=你的Key
REM 仅日志（你已有 filtered_maintenance_data.csv）
python build_index.py --logs data/filtered_maintenance_data.csv
REM 若有 MinerU 导出的手册 .txt
python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv
python query_rag.py -q "你的问题"

仅日志检索（BM25 + 向量 -> RRF -> rerank 阈值过滤）：
python query_log_rag.py -q "某机型某故障现象是否会影响业务" --threshold 0.7 --final-topk 10
若阈值后候选不足，启用自动补齐兜底：
python query_log_rag.py -q "某机型某故障现象是否会影响业务" --threshold 0.7 --final-topk 10 --fill-shortage

====================
手册处理推荐流水线（PDF -> txt -> 索引）
====================

1) 推荐主流程（MinerU）
- 先用 MinerU 把 PDF 转为 txt/md（命令按你本地 MinerU 版本为准）
- 将转换产物放到目录，例如：data/manual_converted/mineru_out

2) 跑统一整理 + 质检 + 建索引
python manual_pipeline.py --converted-dir data/manual_converted/mineru_out

3) 查看质检报告
- data/manual_qc/report.json
- 若某些文件 suspicious=true，建议仅对这些文件用 PaddleOCR 重跑后覆盖 converted 目录

4) 仅整理与质检（不触发 build_index）
python manual_pipeline.py --converted-dir data/manual_converted/mineru_out --skip-build-index

5) 手动建索引（可选）
python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv

6) 一体化执行（PDF -> MinerU -> manual_pipeline）
python run_mineru_pipeline.py \
  --pdf-dir data/manual_pdf \
  --mineru-cmd-template "mineru -i {input} -o {output}"

说明：
- MinerU 各版本 CLI 参数不完全一致，故使用 --mineru-cmd-template 适配
- 模板中必须包含 {input} 和 {output}
- 若只想做 PDF 解析，不触发后续流程：加 --skip-manual-pipeline

说明：
- 本项目 manual 入口目前接收纯文本（.txt）；manual_pipeline 会自动把 .md/.txt 统一为 data/manual_txt/*.txt
- Qdrant 物理落盘目录由 Qdrant 启动配置决定，不由 build_index.py --index-dir 决定

====================
Qdrant 本地启动（离线二进制）
====================

若你本地已有 qdrant 安装包（例如 `/root/autodl-tmp/qdrant-x86_64-unknown-linux-musl.tar.gz`）：

1) 解压
mkdir -p /root/autodl-tmp/qdrant-bin
tar -xzf /root/autodl-tmp/qdrant-x86_64-unknown-linux-musl.tar.gz -C /root/autodl-tmp/qdrant-bin

2) 启动（复用本项目向量存储目录）
QDRANT__STORAGE__STORAGE_PATH="/root/autodl-tmp/device_rag_llm/data/qdrant_storage" \
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
"/root/autodl-tmp/qdrant-bin/qdrant"

3) 校验服务是否可用
curl http://localhost:6333/collections

4) 再运行日志检索
python query_log_rag.py -q "你的问题" --threshold 0.7 --final-topk 10 --fill-shortage