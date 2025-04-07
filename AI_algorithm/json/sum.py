import json

# 假设你的 JSON 数据存储在一个文件中，或者你可以直接将其作为字符串加载
# 如果数据在文件中，请确保文件路径正确
file_path = "data_raw.json"

# 加载 JSON 数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    

# 确保数据是一个列表（JSON 数组）
if isinstance(data, list):
    # 统计样本数
    sample_count = len(data)
    print(f"数据集的样本总数为: {sample_count}")
else:
    print("数据集不是有效的 JSON 数组格式！")