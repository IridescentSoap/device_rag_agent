import pandas as pd

df = pd.read_csv("data/filtered_maintenance_data.csv")

# 空值读入为 float(NaN)，需先当字符串再匹配，否则 "键盘" in device 会报错
device_str = df["device"].fillna("").astype(str)

keywords = ["键盘", "鼠标", "外设", "手机", "耳机"]
pattern = "|".join(keywords)
mask_drop = device_str.str.contains(pattern, regex=True, na=False)

count = int(mask_drop.sum())
df = df[~mask_drop].reset_index(drop=True)

# 生成有序 case_id，固定宽度便于排序和检索
df["case_id"] = [f"case_{i:05d}" for i in range(1, len(df) + 1)]

df.to_csv("data/filtered_maintenance_data.csv", index=False, encoding="utf-8-sig")
print(f"已删除 {count} 行，已写入 case_id 列")
