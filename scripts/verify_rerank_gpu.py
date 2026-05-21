#!/usr/bin/env python3
"""对比 Rerank 在 CPU / GPU 上的设备与耗时（各精排 16 条候选）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from rag.config import INDEX_DIR
from rag.index_store import DualIndexStore
from rag.rerank import (
    _QWEN3_RERANK_CACHE,
    _load_qwen3_rerank_bundle,
    _local_rerank_device,
    _local_rerank_path,
    rerank_chunks_with_scores,
)


def _gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    return f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB"


def _bench_rerank(query: str, items, device: str) -> tuple[float, str, float]:
    _QWEN3_RERANK_CACHE.clear()
    import os

    os.environ["RAG_LOCAL_RERANK_DEVICE"] = device
    bundle = _load_qwen3_rerank_bundle(_local_rerank_path(), device)
    dev = (
        str(next(bundle["model"].parameters()).device)
        if bundle
        else "load_failed"
    )
    t0 = time.perf_counter()
    hits = rerank_chunks_with_scores(query, items, top_n=8)
    elapsed = time.perf_counter() - t0
    top_score = float(hits[0].score) if hits else 0.0
    return elapsed, dev, top_score


def run_cpu_only() -> int:
    store = DualIndexStore.load(INDEX_DIR)
    items = store.log.chunks[:16]
    query = "某设备黑屏是否影响业务，应该如何处理？"
    t, dev, score = _bench_rerank(query, items, "cpu")
    print(f"CPU_RESULT:{t:.4f}|{dev}|{score:.4f}")
    return 0 if score > 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_only:
        sys.exit(run_cpu_only())

    print("=== Rerank GPU 验证 ===")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f".env -> _local_rerank_device(): {_local_rerank_device()}")
    dtype_env = __import__("os").environ.get("RAG_QWEN3_RERANK_DTYPE", "(default float16 on cuda)")
    print(f"RAG_QWEN3_RERANK_DTYPE: {dtype_env}")

    store = DualIndexStore.load(INDEX_DIR)
    items = store.log.chunks[:16]
    query = "某设备黑屏是否影响业务，应该如何处理？"

    print("\n--- 1) GPU 精排 ---")
    print(f"GPU mem (before): {_gpu_mem_mb()}")
    t_gpu, dev_gpu, score_gpu = _bench_rerank(query, items, "cuda")
    print(f"model.device: {dev_gpu}")
    print(f"rerank 16->8 耗时: {t_gpu:.2f}s")
    print(f"top1 score: {score_gpu:.4f}")
    print(f"GPU mem (after): {_gpu_mem_mb()}")

    import subprocess

    print("\n--- 2) CPU 精排（独立子进程） ---")
    r = subprocess.run(
        [sys.executable, str(Path(__file__)), "--cpu-only"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    t_cpu, dev_cpu, score_cpu = -1.0, "error", 0.0
    for line in (r.stdout or "").splitlines():
        if line.startswith("CPU_RESULT:"):
            t_s, dev_cpu, score_s = line[len("CPU_RESULT:") :].split("|", 2)
            t_cpu, score_cpu = float(t_s), float(score_s)
            break
    if r.returncode != 0:
        print(r.stderr or r.stdout)
    print(f"model.device: {dev_cpu}")
    print(f"rerank 16->8 耗时: {t_cpu:.2f}s")
    print(f"top1 score: {score_cpu:.4f}")

    print("\n=== 结论 ===")
    ok_gpu = "cuda" in dev_gpu and score_gpu > 0
    print(f"GPU 精排有效: {ok_gpu} (device={dev_gpu}, top1={score_gpu:.4f})")
    if t_cpu > 0 and t_gpu > 0:
        print(f"加速比 CPU/GPU: {t_cpu / t_gpu:.1f}x")
    if not ok_gpu:
        sys.exit(1)


if __name__ == "__main__":
    main()
