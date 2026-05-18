"""
使用 MinerU（LlamaIndex MinerUReader）将 PDF 转为 TXT。

若输出目录中已存在与 PDF 同主文件名的 .txt 且非空，则跳过转换。
默认输出到项目下 data/manual_txt；可用 --output-dir 改为例如 /data/manual_txt。

Flash 模式下单次请求最多 20 页：页数超过 20 时会自动按「1-20」「21-40」… 分批请求并拼接；
需要 OCR/公式等可用 --mode precision（需 MINERU_TOKEN，单文件上限更高）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from llama_index.readers.mineru import MinerUReader
from llama_index.readers.mineru import base as mineru_base

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "manual_txt"

# MinerU Flash API：单次请求最多 20 页（整文件不含 page_range 时超限即报错）
FLASH_API_MAX_PAGES_PER_REQUEST = 20


def _pdf_num_pages(pdf_path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf_path)).pages)


def _flash_extract_in_page_batches(reader: MinerUReader, src: str, num_pages: int) -> str:
    """Flash 模式：按页范围分批解析（每批最多 FLASH_API_MAX_PAGES_PER_REQUEST 页）。"""
    ranges: list[str] = []
    step = FLASH_API_MAX_PAGES_PER_REQUEST
    for start in range(1, num_pages + 1, step):
        end = min(start + step - 1, num_pages)
        ranges.append(f"{start}-{end}")

    parts: list[str] = []
    saved_pages = reader.pages
    try:
        it = _progress_bar(
            ranges,
            desc=f"MinerU Flash 分批(每批≤{step}页)",
            total=len(ranges),
            unit="批",
        )
        for rng in it:
            reader.pages = rng
            result = reader._extract(src, use_page_range=True)
            reader._check_result(result, src, None)
            parts.append(result.markdown)
    finally:
        reader.pages = saved_pages

    return "\n\n".join(parts)


def expected_txt_path(pdf: Path, out_dir: Path) -> Path:
    return out_dir / f"{pdf.stem}.txt"


def should_skip(pdf: Path, out_dir: Path) -> bool:
    t = expected_txt_path(pdf, out_dir)
    return t.is_file() and t.stat().st_size > 0


def _progress_bar(
    iterable,
    desc: str,
    total: int | None = None,
    unit: str | None = None,
):
    if tqdm is not None:
        u = unit or ("页" if total and total > 1 else "步")
        return tqdm(iterable, desc=desc, total=total, unit=u)
    # 无 tqdm 时简单打印
    class _Dummy:
        def __init__(self, it, **kw):
            self._it = iter(it)
            self._n = 0
            self._total = total or "?"

        def __iter__(self):
            return self

        def __next__(self):
            self._n += 1
            try:
                x = next(self._it)
            except StopIteration:
                raise
            print(f"[{desc}] {self._n}/{self._total}", flush=True)
            return x

    return _Dummy(iterable, desc=desc)


def pdf_to_markdown_with_progress(reader: MinerUReader, pdf_path: Path) -> str:
    """与 MinerUReader.load_data 等价，但在逐页解析时显示进度。"""
    src = str(pdf_path.resolve())
    if reader.split_pages and mineru_base._looks_like_pdf(src):
        download_tmp: TemporaryDirectory | None = None
        split_tmp: TemporaryDirectory | None = None
        try:
            if mineru_base._is_url(src):
                download_tmp, local_path = mineru_base._download_url_to_temp(src)
            else:
                local_path = Path(src)
                if not local_path.exists():
                    raise FileNotFoundError(f"PDF not found: {src}")

            target_pages = (
                mineru_base._parse_page_range(reader.pages) if reader.pages else None
            )
            split_tmp, page_files = mineru_base._split_pdf_to_pages(
                local_path, target_pages
            )
            parts: list[str] = []
            total = len(page_files)
            for page_number, page_path in _progress_bar(
                page_files, desc="MinerU 逐页", total=total
            ):
                result = reader._extract(str(page_path), use_page_range=False)
                reader._check_result(result, src, page_number)
                parts.append(result.markdown)
            return "\n\n".join(parts)
        finally:
            if split_tmp is not None:
                split_tmp.cleanup()
            if download_tmp is not None:
                download_tmp.cleanup()

    # Flash：未指定 pages 且页数 >20 时，整文件调用会报错，改为自动分页范围分批
    if (
        not reader.split_pages
        and reader.mode == "flash"
        and not reader.pages
        and mineru_base._looks_like_pdf(src)
    ):
        n = _pdf_num_pages(pdf_path)
        if n > FLASH_API_MAX_PAGES_PER_REQUEST:
            return _flash_extract_in_page_batches(reader, src, n)

    # 整本一次解析（单步进度；Flash 且 ≤20 页，或 precision 等）
    if tqdm is not None:
        with tqdm(total=1, desc="MinerU 整本解析", unit="步") as bar:
            docs = reader.load_data(src)
            bar.update(1)
    else:
        print("MinerU 整本解析...", flush=True)
        docs = reader.load_data(src)
    return "\n\n".join(d.text for d in docs)


def collect_pdfs(pdf: Path | None, pdf_dir: Path | None) -> list[Path]:
    if pdf is not None:
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF 不存在: {pdf}")
        return [pdf.resolve()]
    if pdf_dir is not None:
        if not pdf_dir.is_dir():
            raise FileNotFoundError(f"目录不存在: {pdf_dir}")
        paths = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
        return [p.resolve() for p in paths]
    raise ValueError("请指定 --pdf 或 --pdf-dir")


def main() -> None:
    p = argparse.ArgumentParser(description="PDF → TXT（MinerU），已存在同名 txt 则跳过")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path, default=None, help="单个 PDF 文件")
    src.add_argument("--pdf-dir", type=Path, default=None, help="目录内所有 .pdf")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"TXT 输出目录（默认: {DEFAULT_OUT}）",
    )
    p.add_argument("--split-pages", action="store_true", help="按页调用 API（可显示逐页进度）")
    p.add_argument(
        "--mode",
        choices=("flash", "precision"),
        default="flash",
        help="MinerU 模式：flash 无需 token；precision 需 MINERU_TOKEN",
    )
    args = p.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.pdf, args.pdf_dir)

    reader = MinerUReader(mode=args.mode, split_pages=args.split_pages)

    for i, pdf_path in enumerate(pdfs, start=1):
        target_txt = expected_txt_path(pdf_path, out_dir)
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}", flush=True)

        if should_skip(pdf_path, out_dir):
            print(f"  已存在非空 TXT，跳过: {target_txt}", flush=True)
            continue

        print(f"  转换中 → {target_txt}", flush=True)
        try:
            text = pdf_to_markdown_with_progress(reader, pdf_path)
        except Exception as e:
            print(f"  失败: {e}", file=sys.stderr, flush=True)
            raise

        target_txt.write_text(text, encoding="utf-8")
        print(f"  已写入 {len(text)} 字符", flush=True)


if __name__ == "__main__":
    main()
