#蒙特卡洛估计A,B规律
import random
from collections import Counter

from matplotlib import pyplot as plt
from tqdm import tqdm

from AI_algorithm.tool.tool import deal_cards_tool, simulate_insertion_tool

if __name__ == '__main__':
    num_simulations_ab = 2_000_0000  # 2000万次A,B的生成


    all_removal_lengths1 = []
    all_removal_lengths2 = []
    all_removal_lengths3 = []

    print(f"开始进行 {num_simulations_ab} 次A,B组合生成及模拟...")
    # 使用tqdm显示进度
    for _ in tqdm(range(num_simulations_ab), desc="模拟进度"):
        A, B = deal_cards_tool()

        card = random.choice(B)

        pos=random.randint(0,6)
            # 调用模拟插入工具
        # 我们只关心 removal_length
        _, _, new_length1, _, newA = simulate_insertion_tool(A, card, pos)
        all_removal_lengths1.append(new_length1)

        B.remove(card)
        card = random.choice(B)
        pos = random.randint(0, len(newA))

        _, _, new_length2, _, newA = simulate_insertion_tool(newA, card, pos)
        all_removal_lengths2.append(new_length2)

        B.remove(card)
        card = random.choice(B)
        pos = random.randint(0, len(newA))

        _, _, new_length3, _, _ = simulate_insertion_tool(newA, card, pos)
        all_removal_lengths3.append(new_length3)


    print(f"\n总共执行了 {len(all_removal_lengths1)} 次 simulate_insertion_tool 调用。")

    # 1. 统计 removal_length 的分布情况
    length_counts = Counter(all_removal_lengths1)
    total_observations = len(all_removal_lengths1)

    print("\n Length 分布情况:")
    # 按 removal_length 排序打印
    sorted_lengths = sorted(length_counts.items())

    for length, count in sorted_lengths:
        percentage = (count / total_observations) * 100
        print(f"   len(A1)( {length}: {count} 次, 占比: {percentage:.2f}%")


    expected_value = sum(all_removal_lengths1) / total_observations
    print(f"\nlen(A1) 的期望值: {expected_value:.4f}")
##########################################
    # 1. 统计 removal_length 的分布情况
    length_counts = Counter(all_removal_lengths2)
    total_observations = len(all_removal_lengths2)

    print("\n len(A2)分布情况:")
    # 按 removal_length 排序打印
    sorted_lengths = sorted(length_counts.items())

    for length, count in sorted_lengths:
        percentage = (count / total_observations) * 100
        print(f"   len(A2) {length}: {count} 次, 占比: {percentage:.2f}%")

    # 2. 计算 removal_length 的期望值
    # E[X] = Σ [x * P(x)]  或者  sum(all_values) / num_values
    expected_value = sum(all_removal_lengths2) / total_observations
    print(f"\n len(A2) 的期望值: {expected_value:.4f}")
##########################################
    # 1. 统计 removal_length 的分布情况
    length_counts = Counter(all_removal_lengths3)
    total_observations = len(all_removal_lengths3)

    print("\n len(A3) 分布情况:")
    # 按 removal_length 排序打印
    sorted_lengths = sorted(length_counts.items())

    for length, count in sorted_lengths:
        percentage = (count / total_observations) * 100
        print(f"  len(A3) {length}: {count} 次, 占比: {percentage:.2f}%")

    # 2. 计算 removal_length 的期望值
    # E[X] = Σ [x * P(x)]  或者  sum(all_values) / num_values
    expected_value = sum(all_removal_lengths3) / total_observations
    print(f"\n len(A3)的期望值: {expected_value:.4f}")