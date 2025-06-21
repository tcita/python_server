import json
import itertools


def generate_permutation_training_data():
    """
    生成基于给定样本的所有排列组合的训练数据集
    A的全排列 × B的全排列 = 6! × 3! = 720 × 6 = 4320 条样本
    """

    # 原始样本数据
    original_A = [11, 13, 3, 10, 12, 6]  # A数组的原始排列
    original_B = [13, 8, 1]  # B数组的原始排列
    max_score = 66  # 最大分数（常数）
    original_best_moves = [[1, 2], [2, 2], [0, 8]]  # 原始最佳移动策略

    dataset = []  # 用于存储所有生成的样本
    sample_count = 0  # 样本计数器

    # 生成A数组的所有排列（6! = 720种）
    for A_perm in itertools.permutations(original_A):
        A_list = list(A_perm)  # 将元组转换为列表格式

        # 生成B数组的所有排列（3! = 6种）
        for B_perm in itertools.permutations(original_B):
            B_list = list(B_perm)  # 将元组转换为列表格式

            # 创建新的样本数据
            # 注意：max_score和best_moves保持不变，因为它们是"数学等价"的常数
            sample_data = {
                "A": A_list,  # A数组的当前排列
                "B": B_list,  # B数组的当前排列
                "max_score": max_score,  # 最大分数（保持不变）
                "best_moves": original_best_moves  # 最佳移动策略（保持不变）
            }

            dataset.append(sample_data)
            sample_count += 1

            # 每处理1000个样本输出一次进度
            if sample_count % 1000 == 0:
                print(f"已生成 {sample_count} 个样本")

    print(f"总共生成了 {sample_count} 个样本")
    print(
        f"预期样本数量: {len(list(itertools.permutations(original_A))) * len(list(itertools.permutations(original_B)))}")

    # 将数据集保存为JSON文件
    with open("json/permutation_dataset.json", "w", encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("数据集已保存到 json/permutation_dataset.json")
    return dataset


# 执行函数生成训练数据
if __name__ == "__main__":
    generate_permutation_training_data()