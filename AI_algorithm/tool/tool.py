import os
import pickle
import random
import numpy as np
import torch

def strategy_TrueScore(A, B, strategy):
    try:
        # 第一步插入
        score1, _, _, _, A = simulate_insertion_tool(
            A,
            B[strategy[0][0]],
            min(len(A), strategy[0][1]))

        # print(f"1. 将 {B[strategy[0][0]]} 插入A<{strategy[0][1]}>号位  得分: {score1}    A变为{A}")

        # 第二步插入
        score2, _, _, _, A = simulate_insertion_tool(
            A,
            B[strategy[1][0]],
            min(len(A),strategy[1][1])
        )
        # print(f"2. 将 {B[strategy[1][0]]} 插入A<{strategy[1][1]}>号位  得分: {score2}    A变为{A}")

        # 第三步插入
        score3, _, _, _, A = simulate_insertion_tool(
            A,
            B[strategy[2][0]],
           min(len(A),strategy[2][1]))

        # print(f"3. 将 {B[strategy[2][0]]} 插入A<{strategy[2][1]}>号位  得分: {score3}    A变为{A}")

        return score1 + score2 + score3

    except IndexError as e:

        print("相关变量的值：")
        print(f"strategy: {strategy}")
        print(f"A: {A}")
        print(f"B: {B}")
def simulate_insertion_tool(A, b, pos):
    """
    将元素b插入到列表A的指定位置pos，并处理可能发生的匹配情况。

    参数:
        A: list - 当前的牌列表(动态变化)
        b: int - 要插入的元素(来自B列表)
        pos: int - b要插入到A中的位置索引

    返回值:
        tuple - 包含5个元素:
            1. int: 本次操作产生的得分
            2. int: 被移除的匹配序列长度
            3. int: 操作后A的长度
            4. int: 是否产生匹配(1=是，0=否)
            5. list: 操作后的A列表
    """
    # 创建A的副本避免修改原始列表
    copy_A = A.copy()

    # 当A为空时,直接往A中插入b,这时候不产生匹配
    if len(copy_A) == 0:
        copy_A.append(b)

        return 0, 0, 1, 0, copy_A

    try:
        # 使用列表的insert函数插入b到A的pos位置
        copy_A.insert(pos, b)

        # 向左右分别寻找b是否与当前列表A中已经存在的数值相同

        left_idx, right_idx = None, None

        # 从插入位置pos的左侧向左遍历列表copy_A，寻找与插入值b相等的最近的元素的索引
        for i in range(pos - 1, -1, -1):
            if copy_A[i] == b:
                left_idx = i
                break

        # 从插入位置pos的右侧向右遍历列表copy_A，寻找与插入值b相等的最近的元素的索引
        for j in range(pos + 1, len(copy_A)):
            if copy_A[j] == b:
                right_idx = j
                break

        # 无论向左还是向右都找不到匹配
        if left_idx is None and right_idx is None:
            return 0, 0, len(copy_A), 0, copy_A

        # 计算插入位置pos与匹配元素的索引(left_idx, right_idx)之间的距离
        # 如果没有匹配的元素，则将距离设置为正无穷大
        left_distance = pos - left_idx if left_idx is not None else float('inf')
        right_distance = right_idx - pos if right_idx is not None else float('inf')

        # 确定匹配发生在哪一侧（左侧或右侧）获取匹配序列的起始和结束索引
        if left_distance < right_distance:

            start, end = left_idx, pos
        elif right_distance < left_distance:

            start, end = pos, right_idx
        else:
            # 两侧距离相等但都不是无穷大（即两侧都有匹配），这种情况不会出现,这是由A在初始化时元素不可重复保证的
            # 如果左右距离都是无穷大，说明没有匹配 根据 IEEE 754 浮点数标准，所有正无穷大 (float('inf')) 被视为相等
            return 0, 0, len(copy_A), 0, copy_A

        # 通过索引获取匹配序列
        removal_interval = copy_A[start:end + 1]
        # 计分
        score = sum(removal_interval)
        # 将A中的匹配序列进行移除
        new_A = copy_A[:start] + copy_A[end + 1:]

        return score, len(removal_interval), len(new_A), 1, new_A

    except Exception as e:
        print(f"当将 {b} 插入到当前A：{A}的位置 {pos} 时发生错误。")
        print(f"Error message: {str(e)}")
        raise
def init_deck_tool():
    return [i for i in range(1, 14) for _ in range(4)]


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
    # 第一张牌必须是与A中某张牌数字相同的牌
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

def complete_best_moves(best_moves):

    if len(best_moves) == 3:
        return best_moves
    # 提取已有的第一个元素
    existing_first_elements = {move[0] for move in best_moves}

    # 需要填充的第一个元素（确保 0, 1, 2 都存在）
    missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]

    # 填充 best_moves 至少 3 个子数组
    while len(best_moves) < 3:
        first_element = missing_first_elements.pop(0)
        best_moves.append([first_element, 0])
    return best_moves


# 像玩家一样做决策
# 添加未来得分计算缓存
_future_score_cache = {}


def calculate_future_score(A, remaining_B):
    # 缓存键
    cache_key = (tuple(A), tuple(remaining_B))

    # 检查缓存
    if cache_key in _future_score_cache:
        return _future_score_cache[cache_key]

    future_score = 0  # 初始化未来得分
    if len(remaining_B) == 0:
        return future_score  # 如果没有剩余的B玩家的牌，返回0
    if not set(A) & set(remaining_B):  # 如果 A 和 B 没有任何重复元素
        return future_score
    # 复制A玩家的牌以便模拟
    simulated_B = remaining_B.copy()

    if len(simulated_B) == 1:
        card_of_B = simulated_B[0]  # 获取最后一张B玩家的牌
        matched = card_of_B  in A  # 判断是否能匹配
        if not matched:
            return future_score  # 如果不能匹配，返回0
        else:
            _, score = get_best_insertion_score(A, card_of_B)
            return score



    elif len(remaining_B) == 2:
        card1, card2 = remaining_B  # 获取最后两张B玩家的牌

        matched1 = card1  in A  # 判断第一张卡是否可以匹配
        matched2 = card2  in A  # 判断第二张卡是否可以匹配

        if card1==card2 and (not (card1 in A)):
            return card1+card2+sum(A)  # 如果两张卡都一样， 一张插最前面一张插最后面

        if not matched1 and (not matched2):
            return future_score  # 如果两张卡都不能匹配，返回0

        if matched1 and not matched2:  # 第一张可以匹配且第二张不能匹配

            _, score1 = get_best_insertion_score(A, card1)
            return score1 + card2  # 第二张可以插到第一张的匹配里

        if matched2 and not matched1: # 第二张可以匹配且第一张不能匹配
            _, score2 = get_best_insertion_score(A, card2)
            return score2 + card1  # 第一张可以插到第二张的匹配里

        # 如果两张卡都能匹配，分次插入，并返回较大得分

        # 先插入card1计算得分
        bestpos, score_card1 = get_best_insertion_score(A, card1)
        # 插入card1后的A变成了什么

        _, _, _, _, newA = simulate_insertion_tool(A, card1, bestpos)
        # 把card2插入新的A
        _, score_card2 = get_best_insertion_score(newA, card2)

        # 计算它们的和

        best_score1 = score_card1 + score_card2

        # 先插入card2计算得分
        bestpos, score_card2 = get_best_insertion_score(A, card2)
        # 插入card2后的A变成了什么
        _, _, _, _, newA = simulate_insertion_tool(A, card2, bestpos)
        # 把card1插入新的A
        _, score_card1 = get_best_insertion_score(newA, card1)

        # 计算它们的和

        best_score2 = score_card1 + score_card2

        return max(best_score2, best_score1)

        # 缓存结果
    _future_score_cache[cache_key] = future_score
    return future_score
def get_best_insertion_score(A, card):
    max_score = -float('inf')
    best_pos = -1

    # 遍历所有可能的插入位置 (从位置0开始)
    for pos in range(0, len(A) + 1):
        score, _, _, _, _ = simulate_insertion_tool(A, card, pos)

        # 找到最大的得分
        if score > max_score:
            max_score = score
            best_pos = pos

    return best_pos, max_score

def calculate_score_by_strategy(A, B, strategy):


    score1, _, _, _,A  = simulate_insertion_tool(A, B[strategy[0][0]],  max(1, min(len(A), strategy[0][1])))
    score2, _, _, _,A  = simulate_insertion_tool(A, B[strategy[1][0]],  max(1, min(len(A), strategy[1][1])))
    score3, _, _, _,_  = simulate_insertion_tool(A, B[strategy[2][0]],  max(1, min(len(A), strategy[2][1])))
    return score1 + score2 + score3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# if __name__ == '__main__':
#
#     A= [13, 8, 11, 4, 6, 7]
#     B= [13, 7, 10]
#     score1, _, _, _,A1  = simulate_insertion_tool(A, B[1], 1)
#     print(A1)
#     print(score1)
#     score2, _, _, _,A2  = simulate_insertion_tool(A1, B[2], 1)
#     print(A2)
#     print(score2)
#     score3, _, _, _,A3  = simulate_insertion_tool(A2, B[0], 3)
#     print(A3)
#     print(score3)

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
