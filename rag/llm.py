"""OpenAI 兼容接口调用生成模型（如 DashScope：qwen-plus / qwen2.5-7b-instruct 等）。"""

from __future__ import annotations

import os

from openai import OpenAI

from rag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def get_client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM API Key，请设置环境变量 LLM_API_KEY（或 DASHSCOPE_API_KEY / OPENAI_API_KEY）"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    client = get_client()
    model = model or LLM_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
