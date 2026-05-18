"""Manual data pipeline: normalize text, run QC, optionally build index.

Expected upstream:
- MinerU or PaddleOCR has converted PDFs into .txt/.md files.

This script focuses on the stable part of the pipeline:
1) Collect converted text files from one directory
2) Normalize and export to data/manual_txt/*.txt
3) Generate QC report to help identify bad OCR files
4) Optionally run build_index.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")
_GARBLED_CHAR_RE = re.compile(r"[\uFFFD]|[�]")


@dataclass
class TextQuality:
    file: str
    chars: int
    lines: int
    blank_line_ratio: float
    garbled_ratio: float
    cjk_ratio: float
    suspicious: bool
    reason: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    # Keep paragraph boundaries while normalizing spaces.
    lines = []
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        lines.append(_WS_RE.sub(" ", ln).rstrip())
    out = "\n".join(lines)
    out = _BLANK_RE.sub("\n\n", out).strip()
    return out + "\n" if out else ""


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text), 1)


def _quality(path: Path, text: str) -> TextQuality:
    lines = text.splitlines() or [""]
    blank_lines = sum(1 for ln in lines if not ln.strip())
    garbled = len(_GARBLED_CHAR_RE.findall(text))
    cjk_ratio = _cjk_ratio(text)
    chars = len(text)
    blank_ratio = blank_lines / max(len(lines), 1)
    garbled_ratio = garbled / max(chars, 1)

    suspicious = False
    reasons: list[str] = []
    if chars < 200:
        suspicious = True
        reasons.append("文本过短(<200字符)")
    if blank_ratio > 0.45:
        suspicious = True
        reasons.append("空行占比过高")
    if garbled_ratio > 0.01:
        suspicious = True
        reasons.append("乱码占比偏高")
    if cjk_ratio < 0.15:
        reasons.append("中文占比较低(可能是英文手册或OCR异常)")

    return TextQuality(
        file=path.name,
        chars=chars,
        lines=len(lines),
        blank_line_ratio=blank_ratio,
        garbled_ratio=garbled_ratio,
        cjk_ratio=cjk_ratio,
        suspicious=suspicious,
        reason="; ".join(reasons),
    )


def _collect_input_files(src_dir: Path) -> list[Path]:
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    return files


def _write_qc_report(report_path: Path, items: list[TextQuality]) -> None:
    payload = {
        "total_files": len(items),
        "suspicious_files": sum(1 for i in items if i.suspicious),
        "details": [
            {
                "file": i.file,
                "chars": i.chars,
                "lines": i.lines,
                "blank_line_ratio": round(i.blank_line_ratio, 4),
                "garbled_ratio": round(i.garbled_ratio, 4),
                "cjk_ratio": round(i.cjk_ratio, 4),
                "suspicious": i.suspicious,
                "reason": i.reason,
            }
            for i in items
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_build_index(project_root: Path, manual_txt_dir: Path, logs_path: Path | None) -> int:
    cmd = ["python", "build_index.py", "--manual-dir", str(manual_txt_dir)]
    if logs_path:
        cmd.extend(["--logs", str(logs_path)])
    return subprocess.call(cmd, cwd=str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted-dir",
        type=str,
        required=True,
        help="MinerU/PaddleOCR 转换产物目录（支持递归读取 .txt/.md）",
    )
    parser.add_argument(
        "--manual-txt-dir",
        type=str,
        default="data/manual_txt",
        help="标准化后的手册文本输出目录",
    )
    parser.add_argument(
        "--qc-report",
        type=str,
        default="data/manual_qc/report.json",
        help="质量报告输出路径(JSON)",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="data/filtered_maintenance_data.csv",
        help="可选：传给 build_index.py 的日志CSV路径",
    )
    parser.add_argument(
        "--skip-build-index",
        action="store_true",
        help="仅做整理与质检，不触发 build_index.py",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    src_dir = (project_root / args.converted_dir).resolve() if not Path(args.converted_dir).is_absolute() else Path(args.converted_dir)
    manual_txt_dir = (project_root / args.manual_txt_dir).resolve() if not Path(args.manual_txt_dir).is_absolute() else Path(args.manual_txt_dir)
    qc_report = (project_root / args.qc_report).resolve() if not Path(args.qc_report).is_absolute() else Path(args.qc_report)
    logs = (project_root / args.logs).resolve() if not Path(args.logs).is_absolute() else Path(args.logs)

    if not src_dir.exists():
        raise FileNotFoundError(f"converted-dir 不存在: {src_dir}")

    files = _collect_input_files(src_dir)
    if not files:
        raise ValueError(f"{src_dir} 下未找到 .txt/.md 文件")

    manual_txt_dir.mkdir(parents=True, exist_ok=True)

    quality_items: list[TextQuality] = []
    for fp in files:
        raw = _read_text(fp)
        norm = _normalize_text(raw)
        out_path = manual_txt_dir / f"{fp.stem}.txt"
        out_path.write_text(norm, encoding="utf-8")
        quality_items.append(_quality(out_path, norm))

    _write_qc_report(qc_report, quality_items)
    print(f"已输出标准化文本: {manual_txt_dir} (共 {len(files)} 个文件)")
    print(f"质检报告: {qc_report}")
    suspicious = [x for x in quality_items if x.suspicious]
    if suspicious:
        print(f"警告: 发现 {len(suspicious)} 个可疑文件，建议回退到 PaddleOCR 复检。")
        for item in suspicious[:10]:
            print(f"  - {item.file}: {item.reason}")

    if args.skip_build_index:
        return

    rc = _run_build_index(project_root, manual_txt_dir, logs if logs.exists() else None)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
