import pickle
import random
import numpy as np



def simulate_insertion_tool(A, x, pos):

    candidate_A = A.copy()
    try :
        if len(candidate_A) == 0:
            candidate_A.append(x)
            return 0, 0, len(candidate_A), 0, candidate_A

        # if pos == 0:
        #     raise ValueError("pos 不能为 0")

        candidate_A.insert(pos, x)
        left_idx, right_idx = None, None

        for i in range(pos - 1, -1, -1):
            if candidate_A[i] == x:
                left_idx = i
                break
    except Exception as e:
        print(f"当将 {x}  插入到当前A： {A}的位置 {pos} 时发生错误。")
        print(f"Error message: {str(e)}")
        raise
    for j in range(pos + 1, len(candidate_A)):
        if candidate_A[j] == x:
            right_idx = j
            break

    if left_idx is None and right_idx is None:
        return 0, 0, len(candidate_A), 0, candidate_A

    left_distance = pos - left_idx if left_idx is not None else float('inf')
    right_distance = right_idx - pos if right_idx is not None else float('inf')

    if left_distance < right_distance:
        start, end = min(pos, left_idx), max(pos, left_idx)
    elif right_distance < left_distance:
        start, end = min(pos, right_idx), max(pos, right_idx)
    else:
        left_sum = sum(candidate_A[min(pos, left_idx):max(pos, left_idx) + 1])
        right_sum = sum(candidate_A[min(pos, right_idx):max(pos, right_idx) + 1])
        start, end = (min(pos, left_idx), max(pos, left_idx)) if left_sum >= right_sum else (
        min(pos, right_idx), max(pos, right_idx))

    removal_interval = candidate_A[start:end + 1]
    score = sum(removal_interval)
    new_A = candidate_A[:start] + candidate_A[end + 1:]

    return score, len(removal_interval), len(new_A), 1, new_A


def init_deck_tool():
    return [i for i in range(1, 14) for _ in range(4)]

 # 不应该使用无法得分的A,B训练
def deal_cards_tool():
    deck = init_deck_tool()
    random.shuffle(deck)
    a_unique = set()
    A = []
    while len(A) < 6:
        card = deck.pop()
        if card not in a_unique:
            a_unique.add(card)
            A.append(card)

    available_A = [card for card in deck if card in a_unique]
    selected = random.choice(available_A)
    deck.remove(selected)  # 从剩余牌中移除选中的牌
    B = [selected]
    # 抽取剩余两张牌
    for _ in range(2):
        B.append(deck.pop())
    return A, B

import pickle

def load_best_genome(filename="../trained/best_genome.pkl"):
    try:
        # 尝试打开并加载文件
        with open(filename, 'rb') as file:
            best_genome = pickle.load(file)
        # print(f"genome loaded from {filename}")
        return best_genome
    except FileNotFoundError:
        # 文件不存在时的处理
        print(f"Error: The file '{filename}' was not found. Please check the file path.")
    except PermissionError:
        # 权限不足时的处理
        print(f"Error: Permission denied when accessing the file '{filename}'.")
    except EOFError:
        # 文件为空或损坏时的处理
        print(f"Error: The file '{filename}' is empty or corrupted.")
    except Exception as e:
        # 捕获其他未知异常
        print(f"An unexpected error occurred while loading the file '{filename}': {e}")


def numsummery(results):
    mean_value = np.mean(results)
    median_value = np.median(results)
    std_dev = np.std(results)
    min_value = np.min(results)
    max_value = np.max(results)
    q1, q3 = np.percentile(results, [25, 75])  # 计算第25百分位数和第75百分位数
    cv = std_dev / mean_value if mean_value != 0 else float('inf')
    print(f"均值: {mean_value}")
    print(f"中位数: {median_value}")
    print(f"标准差: {std_dev}")
    print(f"最小值: {min_value}")
    print(f"最大值: {max_value}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"CV: {cv:.2f}%")

def modsummery(model, num):
    results = []
    for _ in range(num):
        A, B = deal_cards_tool()
        # 裁剪掉负数的情况，transformer有可能出现0
        results.append(max(0, model(A, B)))
    numsummery(results)
def calculate_score_by_strategy(A, B, strategy):


    score1, _, _, _,A  = simulate_insertion_tool(A, B[strategy[0][0]],  max(1, min(len(A), strategy[0][1])))
    score2, _, _, _,A  = simulate_insertion_tool(A, B[strategy[1][0]],  max(1, min(len(A), strategy[1][1])))
    score3, _, _, _,_  = simulate_insertion_tool(A, B[strategy[2][0]],  max(1, min(len(A), strategy[2][1])))
    return score1 + score2 + score3

if __name__ == '__main__':

    A= [13, 8, 11, 4, 6, 7]
    B= [13, 7, 10]
    score1, _, _, _,A1  = simulate_insertion_tool(A, B[1], 1)
    print(A1)
    print(score1)
    score2, _, _, _,A2  = simulate_insertion_tool(A1, B[2], 1)
    print(A2)
    print(score2)
    score3, _, _, _,A3  = simulate_insertion_tool(A2, B[0], 3)
    print(A3)
    print(score3)

# 备用测试代码
# def test_simulate_insertion():
#     A = [1, 2, 3, 3, 2, 1]
#     x = 1
#     for pos in range(1, len(A) + 1):
#         score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(A, x, pos)
#         if match_found:
#             print(f"插入位置: {pos}")
#             print(f"被移除的数组长度: {removed_len}")
#             print(f"更新后的数组: {updated_A}")
#             print("-" * 30)
#
# def test_simulate_insertion2():
#     B = [4,7,5,3,7,1]
#     x = 7
#     for pos in range(1, len(B) + 1):
#         score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(B, x, pos)
#         if match_found:
#             print(f"插入位置: {pos}")
#             print(f"被移除的数组长度: {removed_len}")
#             print(f"更新后的数组: {updated_A}")
#             print("-" * 30)
#
# def test_simulate_insertion3():
#     B = [2,8,1]
#     x = 8
#     for pos in range(1, len(B) + 1):
#         score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(B, x, pos)
#         if match_found:
#             print(f"插入位置: {pos}")
#             print(f"被移除的数组长度: {removed_len}")
#             print(f"更新后的数组: {updated_A}")
#             print("-" * 30)
#
# def Simulate_insertion_Last_Test():
#     best_genome = load_best_genome()
#     A = [2,8,1]
#     B = [8,12]
#     round_score = 0
#     num_moves = 0
#
#     for i, x in enumerate(B):
#         remaining_B = B[i+1:]
#         pos, score, A = genome_choose_insertion(best_genome, A, B, x, remaining_B)
#         print(f"Move {num_moves + 1}: Insert card {x} at position {pos}")
#         print(f"Score: {score}, New A: {A}\n")
#         round_score += score
#         num_moves += 1
#
#     print(f"Final Score: {round_score}, Total Moves: {num_moves}")
