"""
业务标注集评测：检索指标 + 端到端抽样指标。

标注文件（默认 data/eval/business_eval_30.jsonl）每行字段：
  - id, query
  - gold_chunk_ids: 正样本 chunk_id 列表（命中任一即 Recall=1）
  - expected_source: manual | log（用于分源统计）
  - expected_route: manual_heavy | log_heavy | balanced
  - gold_keywords: 端到端答案应覆盖的关键词（自动启发式，非人工打分）

用法：
  python eval_business_rag.py --validate-only
  python eval_business_rag.py
  python eval_business_rag.py --run-e2e --e2e-sample 10
  python eval_business_rag.py --run-e2e --e2e-all
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rag.config import INDEX_DIR, TOPK_BM25, TOPK_VECTOR
from rag.pipeline import DualSourceRagPipeline
from rag.retrieval_modes import mrr, recall_at_k


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "business_eval_30.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "eval" / "reports"


@dataclass
class EvalItem:
    id: str
    query: str
    gold_chunk_ids: list[str]
    expected_source: str
    expected_route: str
    category: str = ""
    gold_keywords: list[str] = field(default_factory=list)


@dataclass
class ItemRetrievalResult:
    id: str
    expected_source: str
    expected_route: str
    predicted_route: str
    route_match: bool
    recall_at: dict[int, float]
    mrr: float
    gold_in_topk: int | None


def load_eval_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    id=obj["id"],
                    query=obj["query"],
                    gold_chunk_ids=list(obj["gold_chunk_ids"]),
                    expected_source=obj.get("expected_source", ""),
                    expected_route=obj.get("expected_route", ""),
                    category=obj.get("category", ""),
                    gold_keywords=list(obj.get("gold_keywords", [])),
                )
            )
    return items


def validate_items(items: list[EvalItem], pipe: DualSourceRagPipeline) -> list[str]:
    errors: list[str] = []
    id_map: dict[str, str] = {}
    for c in pipe.store.manual.chunks:
        id_map[c.chunk_id] = "manual"
    for c in pipe.store.log.chunks:
        id_map[c.chunk_id] = "log"
    for it in items:
        for gid in it.gold_chunk_ids:
            if gid not in id_map:
                errors.append(f"{it.id}: gold_chunk_id 不存在于索引: {gid}")
            elif it.expected_source and id_map[gid] != it.expected_source:
                errors.append(
                    f"{it.id}: gold {gid} 来源为 {id_map[gid]}，与 expected_source={it.expected_source} 不一致"
                )
    return errors


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def eval_retrieval(
    pipe: DualSourceRagPipeline,
    items: list[EvalItem],
    ks: list[int],
    retrieve_kw: dict,
) -> tuple[list[ItemRetrievalResult], dict]:
    results: list[ItemRetrievalResult] = []
    rows_recall: dict[int, list[float]] = {k: [] for k in ks}
    rows_mrr: list[float] = []
    route_hits: list[float] = []
    by_source: dict[str, dict[str, list[float]]] = {}

    for it in items:
        ret = pipe.retrieve(it.query, **retrieve_kw)
        ranked = [h.chunk.chunk_id for h in ret.recall_hits]
        gold = set(it.gold_chunk_ids)
        rec = {k: recall_at_k(gold, ranked, k) for k in ks}
        rr = mrr(gold, ranked)
        route_ok = ret.query_route == it.expected_route if it.expected_route else True
        pos = None
        for i, cid in enumerate(ranked, start=1):
            if cid in gold:
                pos = i
                break

        results.append(
            ItemRetrievalResult(
                id=it.id,
                expected_source=it.expected_source,
                expected_route=it.expected_route,
                predicted_route=ret.query_route,
                route_match=route_ok,
                recall_at=rec,
                mrr=rr,
                gold_in_topk=pos,
            )
        )
        for k in ks:
            rows_recall[k].append(rec[k])
        rows_mrr.append(rr)
        route_hits.append(1.0 if route_ok else 0.0)

        src = it.expected_source or "unknown"
        by_source.setdefault(src, {"recall@5": [], "mrr": [], "route": []})
        by_source[src]["recall@5"].append(rec.get(5, rec.get(max(ks), 0.0)))
        by_source[src]["mrr"].append(rr)
        by_source[src]["route"].append(1.0 if route_ok else 0.0)

    summary = {
        "n": len(items),
        "recall": {f"@{k}": _mean(rows_recall[k]) for k in ks},
        "mrr": _mean(rows_mrr),
        "route_accuracy": _mean(route_hits),
        "by_expected_source": {
            src: {
                "n": len(v["mrr"]),
                "recall@5": _mean(v["recall@5"]),
                "mrr": _mean(v["mrr"]),
                "route_accuracy": _mean(v["route"]),
            }
            for src, v in by_source.items()
        },
    }
    return results, summary


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = answer.lower()
    hit = sum(1 for kw in keywords if kw.lower() in text)
    return hit / len(keywords)


@dataclass
class ItemE2EResult:
    id: str
    citation_hit: bool
    context_gold_in_citations: bool
    keyword_coverage: float
    route_match: bool
    answer_preview: str


def eval_e2e(
    pipe: DualSourceRagPipeline,
    items: list[EvalItem],
    ask_kw: dict,
) -> tuple[list[ItemE2EResult], dict]:
    results: list[ItemE2EResult] = []
    cite_hits: list[float] = []
    kw_covs: list[float] = []
    route_hits: list[float] = []

    for it in items:
        out = pipe.ask(it.query, **ask_kw)
        gold = set(it.gold_chunk_ids)
        cites = set(out.citations)
        cite_hit = bool(gold & cites)
        route_ok = out.query_route == it.expected_route if it.expected_route else True
        kw_cov = _keyword_coverage(out.answer, it.gold_keywords)

        results.append(
            ItemE2EResult(
                id=it.id,
                citation_hit=cite_hit,
                context_gold_in_citations=cite_hit,
                keyword_coverage=kw_cov,
                route_match=route_ok,
                answer_preview=out.answer[:280].replace("\n", " "),
            )
        )
        cite_hits.append(1.0 if cite_hit else 0.0)
        kw_covs.append(kw_cov)
        route_hits.append(1.0 if route_ok else 0.0)

    summary = {
        "n": len(items),
        "citation_recall": _mean(cite_hits),
        "mean_keyword_coverage": _mean(kw_covs),
        "route_accuracy": _mean(route_hits),
    }
    return results, summary


def main() -> None:
    p = argparse.ArgumentParser(description="业务标注集：检索 + 端到端评测")
    p.add_argument("--eval-file", type=str, default=str(DEFAULT_EVAL_FILE))
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--merge-pool-size", type=int, default=80)
    p.add_argument("--final-topk", type=int, default=20)
    p.add_argument("--context-n", type=int, default=6)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--run-e2e", action="store_true", help="运行 LLM 生成评测（较慢，消耗 API）")
    p.add_argument("--e2e-sample", type=int, default=8, help="端到端抽样条数（不含 --e2e-all）")
    p.add_argument("--e2e-all", action="store_true", help="对全部标注条运行端到端")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--detail", action="store_true", help="打印每条失败样本")
    args = p.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        raise SystemExit(f"标注文件不存在: {eval_path}")

    items = load_eval_items(eval_path)
    if len(items) != 30:
        print(f"提示: 当前加载 {len(items)} 条（标准集为 30 条）")

    pipe = DualSourceRagPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    val_errors = validate_items(items, pipe)
    if val_errors:
        print("标注校验失败:")
        for e in val_errors:
            print(" ", e)
        raise SystemExit(1)
    print(f"标注校验通过: {len(items)} 条，manual={sum(1 for i in items if i.expected_source=='manual')} "
          f"log={sum(1 for i in items if i.expected_source=='log')}")

    if args.validate_only:
        return

    retrieve_kw = dict(
        topk_bm25=TOPK_BM25,
        topk_vec=TOPK_VECTOR,
        merge_pool_size=args.merge_pool_size,
        threshold=args.threshold,
        final_topk=args.final_topk,
        fill_shortage=True,
    )
    ask_kw = {**retrieve_kw, "context_n": args.context_n}

    print("\n========== 检索评测（DualSourceRagPipeline.retrieve）==========")
    ret_results, ret_summary = eval_retrieval(pipe, items, args.ks, retrieve_kw)
    print(f"样本数: {ret_summary['n']}")
    print(
        "Recall: "
        + ", ".join(f"{k}={v:.4f}" for k, v in ret_summary["recall"].items())
        + f", MRR={ret_summary['mrr']:.4f}"
    )
    print(f"路由准确率: {ret_summary['route_accuracy']:.4f}")
    for src, st in ret_summary["by_expected_source"].items():
        print(
            f"  [{src}] n={st['n']} Recall@5={st['recall@5']:.4f} "
            f"MRR={st['mrr']:.4f} route={st['route_accuracy']:.4f}"
        )

    e2e_results: list[ItemE2EResult] = []
    e2e_summary: dict | None = None
    if args.run_e2e:
        if args.e2e_all:
            e2e_items = items
        else:
            rng = random.Random(args.seed)
            e2e_items = rng.sample(items, min(args.e2e_sample, len(items)))
        print(f"\n========== 端到端评测（ask，n={len(e2e_items)}）==========")
        e2e_results, e2e_summary = eval_e2e(pipe, e2e_items, ask_kw)
        print(f"引用命中率（gold ∈ citations）: {e2e_summary['citation_recall']:.4f}")
        print(f"关键词覆盖率（启发式）: {e2e_summary['mean_keyword_coverage']:.4f}")
        print(f"路由准确率: {e2e_summary['route_accuracy']:.4f}")

    if args.detail:
        print("\n--- 检索未命中 @5 / 路由错误 ---")
        for r in ret_results:
            if r.recall_at.get(5, 0) < 1.0 or not r.route_match:
                print(
                    f"  {r.id} route={r.predicted_route} expected={r.expected_route} "
                    f"R@5={r.recall_at.get(5, 0):.0f} rank={r.gold_in_topk}"
                )
        if e2e_results:
            print("\n--- 端到端引用未命中 ---")
            for r in e2e_results:
                if not r.citation_hit:
                    print(f"  {r.id} kw_cov={r.keyword_coverage:.2f} | {r.answer_preview[:120]}...")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"business_eval_{ts}.json"
    report = {
        "eval_file": str(eval_path),
        "timestamp": ts,
        "retrieval": ret_summary,
        "retrieval_items": [asdict(r) for r in ret_results],
    }
    if e2e_summary:
        report["e2e"] = e2e_summary
        report["e2e_items"] = [asdict(r) for r in e2e_results]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
