"""
构建双源索引（手册 + 运维日志）。

向量写入 Qdrant；BM25 与 chunk 元数据落在 --index-dir 下。

手册：支持目录/单文件为 .txt；若路径中含 .pdf，会先检查是否已有可用 .txt
（同目录旁挂或 --manual-txt-dir 下同名非空文件）。缺失时可使用 --convert-manual-pdf
调用 MinerU 生成 TXT 后再建索引。
日志：默认 data/filtered_maintenance_data.csv

用法：
  set QDRANT_URL=http://localhost:6333
  python build_index.py --logs data/filtered_maintenance_data.csv
  python build_index.py --manual-dir data/manual_txt --logs data/filtered_maintenance_data.csv --recreate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunking import (
    collect_manual_txt_sources,
    load_manual_from_txt_paths,
)
from rag.config import INDEX_DIR, MANUAL_CHUNK_OVERLAP, MANUAL_CHUNK_SIZE, QDRANT_COLLECTION_PREFIX
from rag.embedder import BgeEmbedder
from rag.index_store import DualIndexStore
from rag.ingest_logs import load_logs_from_csv

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_manual_txt_dir() -> Path:
    return PROJECT_ROOT / "data" / "manual_txt"


def _convert_pdfs_to_manual_txt(
    pdfs: list[Path],
    manual_txt_dir: Path,
    mode: str,
    split_pages: bool,
) -> None:
    """依赖 mineru_to_pdf 与 MinerUReader；产出写入 manual_txt_dir/{stem}.txt。"""
    from llama_index.readers.mineru import MinerUReader

    from mineru_to_pdf import pdf_to_markdown_with_progress

    manual_txt_dir.mkdir(parents=True, exist_ok=True)
    reader = MinerUReader(mode=mode, split_pages=split_pages)
    for pdf in pdfs:
        out = manual_txt_dir / f"{pdf.stem}.txt"
        print(f"  PDF→TXT: {pdf.name} → {out}", flush=True)
        text = pdf_to_markdown_with_progress(reader, pdf)
        out.write_text(text, encoding="utf-8")


def _avg_chunk_length(chunks: list) -> float:
    if not chunks:
        return 0.0
    return sum(len(c.text) for c in chunks) / len(chunks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    p.add_argument("--qdrant-prefix", type=str, default=None, help="Qdrant 集合前缀，默认取环境变量 QDRANT_COLLECTION_PREFIX")
    p.add_argument("--recreate", action="store_true", help="重建 Qdrant 集合（删除同名集合后写入）")
    p.add_argument("--embedding-model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--logs", type=str, default="data/filtered_maintenance_data.csv")
    p.add_argument("--manual-file", type=str, default=None, help="单个手册文件（.txt 或 .pdf）")
    p.add_argument(
        "--manual-dir",
        type=str,
        default=None,
        help="手册目录（*.txt；若有 *.pdf 会先检查是否已有对应 TXT）",
    )
    p.add_argument(
        "--manual-txt-dir",
        type=str,
        default=None,
        help=f"PDF 转换后的 TXT 存放目录（默认: {_default_manual_txt_dir()}）",
    )
    p.add_argument(
        "--convert-manual-pdf",
        action="store_true",
        help="当存在缺少 TXT 的 PDF 时，用 MinerU 转换后再建索引（否则报错退出）",
    )
    p.add_argument(
        "--mineru-mode",
        choices=("flash", "precision"),
        default="flash",
        help="与 mineru_to_pdf 一致；precision 需 MINERU_TOKEN",
    )
    p.add_argument(
        "--mineru-split-pages",
        action="store_true",
        help="MinerU 按页解析（大文档可选）",
    )
    p.add_argument("--chunk-size", type=int, default=MANUAL_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=MANUAL_CHUNK_OVERLAP)
    args = p.parse_args()

    if args.manual_file is not None and args.manual_dir is not None:
        print("错误: --manual-file 与 --manual-dir 请勿同时使用。", file=sys.stderr)
        sys.exit(1)

    from rag import config

    model = args.embedding_model or config.EMBEDDING_MODEL
    embedder = BgeEmbedder(model, device=args.device)

    store = DualIndexStore(Path(args.index_dir))
    prefix = args.qdrant_prefix or QDRANT_COLLECTION_PREFIX
    m_coll, l_coll = store.collection_names(prefix)

    manual_txt_dir = (
        Path(args.manual_txt_dir).resolve()
        if args.manual_txt_dir
        else _default_manual_txt_dir()
    )

    manual_chunks = []
    if args.manual_file or args.manual_dir:
        mf = Path(args.manual_file).resolve() if args.manual_file else None
        md = Path(args.manual_dir).resolve() if args.manual_dir else None
        ready_txt, need_pdf = collect_manual_txt_sources(mf, md, manual_txt_dir)

        if need_pdf:
            if not args.convert_manual_pdf:
                print(
                    "以下 PDF 尚无可用 TXT（已在同目录或 --manual-txt-dir 下查找同名非空 .txt）：",
                    file=sys.stderr,
                )
                for p in need_pdf:
                    print(f"  - {p}", file=sys.stderr)
                print(
                    "\n请先运行 mineru_to_pdf.py 生成 TXT，或添加参数 --convert-manual-pdf",
                    file=sys.stderr,
                )
                sys.exit(1)
            _convert_pdfs_to_manual_txt(
                need_pdf,
                manual_txt_dir,
                args.mineru_mode,
                args.mineru_split_pages,
            )
            ready_txt, need_pdf = collect_manual_txt_sources(mf, md, manual_txt_dir)
            if need_pdf:
                print("转换后仍有 PDF 未得到 TXT:", file=sys.stderr)
                for p in need_pdf:
                    print(f"  - {p}", file=sys.stderr)
                sys.exit(1)

        if not ready_txt:
            print("手册路径下未找到可用于索引的 .txt。", file=sys.stderr)
            sys.exit(1)

        manual_chunks = load_manual_from_txt_paths(
            ready_txt,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        print("未指定 --manual-file 或 --manual-dir，手册索引为空（仅构建日志索引）。")
    print(f"manual_chunks 数量: {len(manual_chunks)}，平均长度: {_avg_chunk_length(manual_chunks):.1f}")

    log_chunks: list = []
    if args.logs and Path(args.logs).exists():
        log_chunks = load_logs_from_csv(args.logs)
        print(f"日志条数: {len(log_chunks)}")
    else:
        print(f"未找到日志文件: {args.logs}，跳过日志索引。")
    print(f"log_chunks 数量: {len(log_chunks)}，平均长度: {_avg_chunk_length(log_chunks):.1f}")

    store.manual.build(manual_chunks, embedder, store.client, m_coll, recreate=args.recreate)
    store.log.build(log_chunks, embedder, store.client, l_coll, recreate=args.recreate)
    store.save()
    print(f"索引元数据已保存: {args.index_dir}")
    print(f"Qdrant 集合: {m_coll}, {l_coll}（URL: {config.QDRANT_URL}）")


if __name__ == "__main__":
    main()
