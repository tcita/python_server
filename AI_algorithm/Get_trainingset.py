import json

from AI_algorithm.brute_force import recursive_strategy



from AI_algorithm.tool.tool import deal_cards_tool


def generate_training_data(num_samples=30000):
    dataset = []

    for _ in range(num_samples):

        # 不应该使用无法得分的A,B训练
        while True:
            A, B = deal_cards_tool()  # 初始A, B   A, B 都是 list<int>

            # 检查 A 和 B 是否有任何重复的元素
            if not set(A) & set(B):  # 如果 A 和 B 没有任何重复元素
                break  # 退出循环，继续处理这对 A, B

        max_score, best_moves = recursive_strategy(A, B)

        # 提取已有的第一个元素
        existing_first_elements = {move[0] for move in best_moves}

        # 需要填充的第一个元素（确保 0, 1, 2 都存在）
        missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]

        # 填充 best_moves 至少 3 个子数组
        while len(best_moves) < 3:
            first_element = missing_first_elements.pop(0)  # 获取缺失的第一个元素
            best_moves.append([first_element, 1])  # 组合 [缺失的元素, 1]



        dataset.append({"A": A, "B": B, "max_score": max_score, "best_moves": best_moves})

    with open("json/data_raw.json", "w") as f:
        json.dump(dataset, f, indent=4)


generate_training_data()
