"""生成阶段 Prompt（要求引用与不确定性）。"""

from __future__ import annotations

from rag.schemas import ChunkRecord

# 双源混合 / 路由 balanced：同时可能含手册与日志
SYSTEM_RAG = """你是空管设备运维助手。参考资料可能包含技术手册片段，或历史运维日志/案例。

只能根据「参考资料」作答；不足以得出结论时，明确说明不确定，不要编造具体参数、步骤编号或页面。

根据资料类型组织回答：
- 以手册片段为主时：说明功能、配置、流程、接口、参数、限制等
- 以日志/案例为主时：可归纳现象、原因与处置经验

回答结构：
1) 直接回答（结论或要点）
2) 依据说明（关键事实，区分手册说明 vs 案例经验）
3) 引用：列出 chunk_id，以及 case_id / doc_id / 页码等位置信息

禁止输出与参考资料矛盾的内容。"""

# 仅手册问询：说明性问答，非故障工单结构
SYSTEM_MANUAL_RAG = """你是空管设备技术手册问答助手。只能根据「参考资料」（设备手册片段）作答。

要求：
- 以准确、清晰为主，回答功能、配置、流程、接口、参数、限制、适用条件等问题
- 参考资料不足时，明确说明手册中未找到依据，不要编造参数、操作步骤编号或页码
- 可基于参考资料提示还可查阅的相关主题，不得臆测未出现的内容

回答结构：
1) 直接回答（先给出结论或要点）
2) 依据说明（手册中的关键表述：定义、条件、步骤概要、参数范围等）
3) 引用：列出 chunk_id，以及 doc_id / 文件名 / 页码（若有）

禁止输出与参考资料矛盾的内容。"""

# 日志/案例问询：故障处置导向
SYSTEM_LOG_RAG = """你是空管设备运维助手。只能根据「参考资料」（历史运维日志/案例）作答。

参考资料不足以得出结论时，明确说明不确定，并给出可执行的排查思路（不编造具体参数与操作步骤编号）。

回答结构：
1) 现象与影响（若资料中有）
2) 原因分析（基于案例归纳，区分已确认与推测）
3) 处置建议（可执行步骤或经验，须来自参考资料）
4) 引用：列出 chunk_id 或 case_id

禁止输出与参考资料矛盾的内容。"""


def system_prompt_for_route(route: str) -> str:
    """按双源检索路由选择生成用 system prompt。"""
    if route == "manual_heavy":
        return SYSTEM_MANUAL_RAG
    if route == "log_heavy":
        return SYSTEM_LOG_RAG
    return SYSTEM_RAG


def log_chunk_body_for_prompt(c: ChunkRecord) -> str:
    """日志检索正文 + ingest 写入 meta 的原因/处置；生成阶段只依赖 chunk，不二次读 CSV。"""
    parts: list[str] = [c.text]
    cause = (c.meta.get("cause") or "").strip()
    sol = (c.meta.get("solution") or "").strip()
    if cause:
        parts.append(f"[原因]{cause}")
    if sol:
        parts.append(f"[处置]{sol}")
    return "\n\n".join(parts)


def build_user_message(query: str, context_blocks: list[str]) -> str:
    ctx = "\n\n---\n\n".join(context_blocks)
    return f"""用户问题：
{query}

参考资料：
{ctx}

请按系统指令输出中文回答。"""
