#!/usr/bin/env python3
"""
Agentic RAG 命令行 Demo

用法（在项目根目录）：
  python scripts/run_agent_demo.py -q "某设备黑屏是否影响业务，应该如何处理？"
  python scripts/run_agent_demo.py -q "..." --retrieve-only
  python scripts/run_agent_demo.py -q "..." --json
  python scripts/run_agent_demo.py -q "..." --fast
  AGENT_FAST_MODE=1 python scripts/run_agent_demo.py -q "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.workflow import AgentWorkflow


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic RAG Demo")
    p.add_argument("-q", "--query", type=str, required=True, help="用户问题")
    p.add_argument(
        "--retrieve-only",
        action="store_true",
        help="仅检索与证据判断，不调用 LLM 生成",
    )
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    p.add_argument("--index-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：缩小候选池与 Rerank 规模（也可用环境变量 AGENT_FAST_MODE=1）",
    )
    p.add_argument(
        "--max-supplement-rounds",
        type=int,
        default=1,
        help="证据不足时的最大补充检索轮数（默认 1）",
    )
    p.add_argument(
        "--llm-planner",
        action="store_true",
        help="启用 LLM Planner（失败自动回退规则 Planner）",
    )
    p.add_argument(
        "--llm-decompose",
        action="store_true",
        help="启用 LLM Query Decomposition（失败自动回退规则拆解）",
    )
    p.add_argument(
        "--llm-rewrite",
        action="store_true",
        help="启用 LLM 多轮追问改写（失败自动回退规则改写）",
    )
    args = p.parse_args()

    from agent.tools import RagTools

    tools = RagTools(
        index_dir=args.index_dir,
        device=args.device,
        fast_mode=args.fast,
    )
    if not tools.pipe.store.manual.chunks and not tools.pipe.store.log.chunks:
        print("错误: 索引为空，请先运行 build_index.py", file=sys.stderr)
        sys.exit(1)

    wf = AgentWorkflow(tools=tools)
    resp = wf.run(
        args.query,
        skip_llm=args.retrieve_only,
        fast_mode=args.fast,
        max_supplement_rounds=args.max_supplement_rounds,
        use_llm_planner=args.llm_planner,
        use_llm_decompose=args.llm_decompose,
        use_llm_rewrite=args.llm_rewrite,
    )
    out = resp.to_dict()

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n--- 回答 ---\n")
    print(resp.answer)


if __name__ == "__main__":
    main()
