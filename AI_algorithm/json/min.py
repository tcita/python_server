import json

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 找到最低 max_score 的样本
def find_lowest_max_score(dataset):
    min_score = float('inf')  # 初始化为正无穷大
    best_sample = None

    for sample in dataset:
        if sample["max_score"] < min_score:
            min_score = sample["max_score"]
            best_sample = sample

    return min_score, best_sample

# 主程序
if __name__ == "__main__":
    # JSON 文件路径（请替换为你的实际文件路径）
    # file_path = "data_raw.json"
    file_path="transformer_error_cases.json"
    # 加载数据集
    dataset = load_json(file_path)

    # 找到最低 max_score 的样本
    min_score, best_sample = find_lowest_max_score(dataset)

    # 输出结果
    if best_sample:
        print(f"数据集样本个数:{len(dataset)}")
        print("最低 max_score:", min_score)
        print("对应的数据:", best_sample)
    else:
        print("数据集为空")