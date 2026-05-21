"""精排：仅使用本地 Reranker。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from rag.config import RERANK_ENABLED, RERANK_MODEL
from rag.schemas import ChunkRecord

_LOCAL_RERANK_CACHE: dict[str, Any] = {}
_QWEN3_RERANK_CACHE: dict[str, Any] = {}
_RERANK_BACKEND_PRINTED = False


def _result_index(item: Any) -> int | None:
    if isinstance(item, dict):
        v = item.get("index")
    else:
        v = getattr(item, "index", None)
    return int(v) if v is not None else None


def _get_results(resp: Any) -> list[Any] | None:
    output = getattr(resp, "output", None)
    if output is None:
        return None
    if isinstance(output, dict):
        r = output.get("results")
        return r if isinstance(r, list) else None
    r = getattr(output, "results", None)
    return r if isinstance(r, list) else None


def _result_score(item: Any) -> float:
    if isinstance(item, dict):
        v = item.get("relevance_score", item.get("score", 0.0))
    else:
        v = getattr(item, "relevance_score", getattr(item, "score", 0.0))
    try:
        return float(v)
    except Exception:
        return 0.0


@dataclass
class RerankHit:
    chunk: ChunkRecord
    score: float


def _local_rerank_path() -> str:
    return os.environ.get(
        "RAG_LOCAL_RERANK_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "reranker" / "Qwen3-Reranker-4B"),
    )


def _local_rerank_device() -> str:
    from rag import config

    v = os.environ.get("RAG_LOCAL_RERANK_DEVICE", "").strip()
    return v or config.RERANK_DEVICE


def _local_rerank_batch_size() -> int:
    try:
        return max(1, int(os.environ.get("RAG_LOCAL_RERANK_BATCH_SIZE", "8")))
    except Exception:
        return 8


def _local_rerank_max_length() -> int:
    try:
        return max(16, int(os.environ.get("RAG_LOCAL_RERANK_MAX_LENGTH", "512")))
    except Exception:
        return 512


def _qwen3_rerank_max_length() -> int:
    """Qwen3-Reranker 官方示例默认 8192，与 CrossEncoder 默认 512 分开配置。"""
    try:
        return max(256, int(os.environ.get("RAG_QWEN3_RERANK_MAX_LENGTH", "8192")))
    except Exception:
        return 8192


def _is_qwen3_causal_rerank(model_path: str) -> bool:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    arch = cfg.get("architectures") or []
    return "Qwen3ForCausalLM" in arch or cfg.get("model_type") == "qwen3"


def _format_qwen3_instruction(instruction: str, query: str, doc: str) -> str:
    return (
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=doc,
        )
    )


def _qwen3_process_inputs(
    tokenizer: Any,
    pairs: list[str],
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    max_length: int,
) -> dict[str, Any]:
    inner_max = max_length - len(prefix_tokens) - len(suffix_tokens)
    inner_max = max(inner_max, 16)
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=inner_max,
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    return tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)


@torch.no_grad()
def _qwen3_compute_logits(
    model: Any,
    inputs: dict[str, Any],
    token_true_id: int,
    token_false_id: int,
) -> list[float]:
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    stacked = torch.stack([false_vector, true_vector], dim=1)
    stacked = torch.nn.functional.log_softmax(stacked, dim=1)
    return stacked[:, 1].exp().tolist()


def _resolve_rerank_device(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _qwen3_rerank_dtype(dev: str) -> "torch.dtype":
    """CUDA 上默认 float16；部分环境 bfloat16 会触发 illegal memory access。"""
    v = os.environ.get("RAG_QWEN3_RERANK_DTYPE", "").strip().lower()
    if v in ("float16", "fp16"):
        return torch.float16
    if v in ("float32", "fp32"):
        return torch.float32
    if v in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_qwen3_rerank_bundle(model_path: str, device: str | None) -> dict[str, Any] | None:
    ml = _qwen3_rerank_max_length()
    dev = _resolve_rerank_device(device)
    cache_key = f"{model_path}::{dev}::{ml}"
    if cache_key in _QWEN3_RERANK_CACHE:
        return _QWEN3_RERANK_CACHE[cache_key]
    if not Path(model_path).exists():
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        dtype = _qwen3_rerank_dtype(dev)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).eval()
        model = model.to(dev)
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        bundle = {
            "model": model,
            "tokenizer": tokenizer,
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "token_true_id": token_true_id,
            "token_false_id": token_false_id,
            "max_length": ml,
            "device": dev,
        }
    except Exception:
        return None
    _QWEN3_RERANK_CACHE[cache_key] = bundle
    return bundle


def _local_rerank_qwen3(
    query: str,
    items: Sequence[ChunkRecord],
    top_n: int,
    model_path: str,
) -> list[RerankHit]:
    bundle = _load_qwen3_rerank_bundle(model_path, _local_rerank_device())
    if not bundle:
        return []
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    prefix_tokens = bundle["prefix_tokens"]
    suffix_tokens = bundle["suffix_tokens"]
    token_true_id = bundle["token_true_id"]
    token_false_id = bundle["token_false_id"]
    max_length = bundle["max_length"]
    device = bundle["device"]

    instruct = os.environ.get(
        "RAG_QWEN3_RERANK_INSTRUCTION",
        "Given a web search query, retrieve relevant passages that answer the query",
    ).strip() or "Given a web search query, retrieve relevant passages that answer the query"

    doc_max = int(os.environ.get("RAG_RERANK_DOC_MAX_CHARS", "8000"))
    texts = [_format_qwen3_instruction(instruct, query, c.text[:doc_max]) for c in items]
    all_scores: list[float] = []
    bs = _local_rerank_batch_size()
    try:
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = _qwen3_process_inputs(
                tokenizer, batch, prefix_tokens, suffix_tokens, max_length
            )
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            all_scores.extend(
                _qwen3_compute_logits(model, inputs, token_true_id, token_false_id)
            )
    except Exception:
        return []

    ranked = sorted(zip(items, all_scores), key=lambda x: float(x[1]), reverse=True)[
        :top_n
    ]
    _print_backend_once(f"[RERANK] backend=qwen3-causalLM path={model_path}")
    return [RerankHit(chunk=c, score=float(s)) for c, s in ranked]


def _print_backend_once(msg: str) -> None:
    global _RERANK_BACKEND_PRINTED
    if _RERANK_BACKEND_PRINTED:
        return
    print(msg)
    _RERANK_BACKEND_PRINTED = True


def _load_local_cross_encoder(model_path: str, device: str | None) -> Any | None:
    cache_key = f"{model_path}::{device or 'auto'}::{_local_rerank_max_length()}"
    if cache_key in _LOCAL_RERANK_CACHE:
        return _LOCAL_RERANK_CACHE[cache_key]
    if not Path(model_path).exists():
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        kwargs: dict[str, Any] = {"max_length": _local_rerank_max_length()}
        if device:
            kwargs["device"] = device
        model = CrossEncoder(model_path, **kwargs)
    except Exception:
        return None
    _LOCAL_RERANK_CACHE[cache_key] = model
    return model


def _local_rerank(
    query: str,
    items: Sequence[ChunkRecord],
    top_n: int,
) -> list[RerankHit]:
    model_path = _local_rerank_path()
    if not Path(model_path).exists():
        return []
    if _is_qwen3_causal_rerank(model_path):
        return _local_rerank_qwen3(query, items, top_n, model_path)

    model = _load_local_cross_encoder(model_path, _local_rerank_device())
    if model is None:
        return []
    pairs = [(query, c.text[:8000]) for c in items]
    try:
        scores = model.predict(
            pairs,
            batch_size=_local_rerank_batch_size(),
            show_progress_bar=False,
        )
    except Exception:
        return []
    ranked = sorted(
        zip(items, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )[:top_n]
    _print_backend_once(f"[RERANK] backend=local path={model_path}")
    return [RerankHit(chunk=c, score=float(s)) for c, s in ranked]


def rerank_chunks_with_scores(
    query: str,
    items: Sequence[ChunkRecord],
    top_n: int,
    model: str | None = None,
) -> list[RerankHit]:
    if not items:
        return []
    model = model or RERANK_MODEL
    if not RERANK_ENABLED:
        _print_backend_once("[RERANK] disabled: return score=0 fallback")
        return [RerankHit(chunk=c, score=0.0) for c in items[:top_n]]

    local_hits = _local_rerank(query, items, top_n)
    if local_hits:
        return local_hits

    _print_backend_once("[RERANK] local backend unavailable: return score=0 fallback")
    return [RerankHit(chunk=c, score=0.0) for c in items[:top_n]]


def rerank_chunks(
    query: str,
    items: Sequence[ChunkRecord],
    top_n: int,
    model: str | None = None,
) -> list[ChunkRecord]:
    hits = rerank_chunks_with_scores(query, items, top_n, model=model)
    return [h.chunk for h in hits]
