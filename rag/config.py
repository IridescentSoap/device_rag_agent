"""默认配置；可通过环境变量覆盖。"""

import os
from pathlib import Path

# 项目根目录（device_rag_llm）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 索引落盘目录（BM25 与 chunks 元数据；向量在 Qdrant）
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", str(PROJECT_ROOT / "data" / "rag_index")))

# Qdrant（向量库）
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
# 集合名前缀，最终为 {prefix}_manual / {prefix}_log
QDRANT_COLLECTION_PREFIX = os.environ.get("QDRANT_COLLECTION_PREFIX", "device_rag")

# Embedding（默认优先本地目录，可通过 RAG_EMBEDDING_MODEL 覆盖）
EMBEDDING_MODEL_LOCAL_DIR = PROJECT_ROOT / "models" / "embedding" / "bge-large-zh-v1.5"
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    str(EMBEDDING_MODEL_LOCAL_DIR),
)

# 生成（OpenAI 兼容，如 DashScope）
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen-plus")
LLM_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
OPENAI_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get(
    "OPENAI_API_KEY", ""
)

# 重排序（DashScope Text ReRank；百炼控制台可查，如 gte-rerank-v2、qwen3-rerank）
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "gte-rerank-v2")
RERANK_ENABLED = True

# 手册分块
MANUAL_CHUNK_SIZE = int(os.environ.get("RAG_MANUAL_CHUNK_SIZE", "1024"))
MANUAL_CHUNK_OVERLAP = int(os.environ.get("RAG_MANUAL_CHUNK_OVERLAP", "128"))

# 检索
TOPK_BM25 = int(os.environ.get("RAG_TOPK_BM25", "20"))
TOPK_VECTOR = int(os.environ.get("RAG_TOPK_VECTOR", "20"))
TOPK_PER_SOURCE = int(os.environ.get("RAG_TOPK_PER_SOURCE", "15"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RERANK_TOP_N = int(os.environ.get("RAG_RERANK_TOP_N", "8"))
FINAL_CONTEXT_N = int(os.environ.get("RAG_FINAL_CONTEXT_N", "8"))
MIN_CONTEXT_N = int(os.environ.get("RAG_MIN_CONTEXT_N", "2"))

# 双源路由：RRF 强度比较 + 语料规模归一化 + 轻量关键词辅助
ROUTE_TOPK_SUM = int(os.environ.get("RAG_ROUTE_TOPK_SUM", "5"))
ROUTE_RATIO = float(os.environ.get("RAG_ROUTE_RATIO", "1.1"))
ROUTE_USE_CORPUS_NORM = os.environ.get("RAG_ROUTE_USE_CORPUS_NORM", "1").lower() in (
    "1",
    "true",
    "yes",
)
ROUTE_KEYWORD_HINT = os.environ.get("RAG_ROUTE_KEYWORD_HINT", "1").lower() in (
    "1",
    "true",
    "yes",
)

# 分源精排：各源独立 rerank 后按路由配额合并（缓解 log 被手册长文本压制）
PER_SOURCE_RERANK = os.environ.get("RAG_PER_SOURCE_RERANK", "1").lower() in (
    "1",
    "true",
    "yes",
)
PER_SOURCE_POOL = int(os.environ.get("RAG_PER_SOURCE_POOL", "60"))
LOG_DEDUP_MAX_PER_SYSTEM = int(os.environ.get("RAG_LOG_DEDUP_MAX_PER_SYSTEM", "3"))
# 分源合并时异源条数上限（log_heavy 时手册最多几条）
MERGE_MAX_CROSS_SOURCE = int(os.environ.get("RAG_MERGE_MAX_CROSS_SOURCE", "1"))
