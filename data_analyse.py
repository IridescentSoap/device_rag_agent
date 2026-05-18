## 
# 数据库type说明：
#  1 - 设备故障
#  2 - 设备环境异常事件
#  3 - 通信干扰
#  8 - 监视信号异常
#  9 - GPS干扰
#  12 - 导航信号异常
#  13 - 机载应答机干扰
#  14 - 通信信号异常
##
import pandas as pd
import json
import os
from openai import OpenAI, BadRequestError

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get(
    "OPENAI_API_KEY", ""
)
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
SYSTEM_PROMPT = """
任务：将输入文本提取为 JSON。
仅输出一个 JSON 对象，不要输出解释。
字段固定为：
{
  "system": "",
  "device": "",
  "falut_description": "",
  "falut_influence": "",
  "falut_solution": "",
  "falut_cause": ""
}
规则：
1) 字段缺失时填空字符串 ""。
2) 保留原文语义，不编造信息。
3) 输出必须是合法 JSON。
"""

def data_analyse(data_path):
    result_list = []
    df = pd.read_csv(data_path)
    for i in range(len(df)):
        record_dict = {}
        if df.iloc[i]['type_id'] == 1:
            record_dict['type'] = '设备故障'
            record_dict['content'] = df.iloc[i]['content']  
            result_list.append(record_dict)
    result_df = pd.DataFrame(result_list)
    result_df.to_csv("data/device_maintenance_data.csv", index=False, encoding="utf-8-sig")
    print("已写入文件: device_maintenance_data.csv")

if __name__ == "__main__":
    # data_analyse("data/duty_log.csv")
    # print("数据分析完成")
    # data_analyse("data/duty_log_history.csv")
    # print("数据分析完成")
    df = pd.read_csv("data/device_maintenance_data.csv")
    json_result_list = []

    for i in range(len(df)):
        try:
            if i < 1175:
                continue
            content = df.iloc[i]["content"]
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                response_format={"type": "json_object"}
            )

            content_json_str = response.choices[0].message.content
            print(content_json_str)
            try:
                content_json = json.loads(content_json_str)
            except json.JSONDecodeError:
                # 如果模型偶发返回非标准 JSON，则保底记录原文，避免流程中断
                content_json = {"raw_content": content_json_str}

            content_json["source_content"] = content
            json_result_list.append(content_json)
            print(f"已处理 {i + 1}/{len(df)}")
        except BadRequestError as e:
            print(f"第 {i + 1} 条数据请求失败，已跳过: {e}")
            continue

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "filtered_maintenance_data.csv")
    pd.DataFrame(json_result_list).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已写入文件: {output_path}")