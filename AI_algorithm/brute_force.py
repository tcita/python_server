
from AI_algorithm.GA import simulate_insertion


from collections import defaultdict
from copy import copy
# 返回的bestmoves中的位置是基于变化后的A（simulate_insertion处理过）的索引
def recursive_StrategyAndScore(A, B, original_indices=None, move_sequence=None, used_indices=None):
    if move_sequence is None:
        move_sequence = []
    if original_indices is None:
        original_indices = defaultdict(list)
        for idx, value in enumerate(B):
            original_indices[value].append(idx)
    if used_indices is None:
        used_indices = defaultdict(int)  # 初始为0

    if not B:
        return 0, move_sequence  # 基础情况返回当前路径

    max_score = 0
    best_moves = []

    for i, b in enumerate(B):
        # 使用当前used_indices的拷贝
        current_used = copy(used_indices)
        if current_used[b] < len(original_indices[b]):
            index_b = original_indices[b][current_used[b]]
            current_used[b] += 1  # 递增当前元素的计数

            for a in range(0, len(A) + 1):
                insertscore, _, _, _, AA = simulate_insertion(A, b, a)
                B_new = B[:i] + B[i+1:]

                # 递归调用，传递拷贝后的used_indices
                score, moves = recursive_StrategyAndScore(
                    AA, B_new, original_indices,
                    move_sequence ,
                    current_used  # 使用拷贝后的used_indices
                )

                new_score = score + insertscore
                if new_score > max_score:
                    max_score = new_score
                    # 确保新的插入操作 (index_b, a) 是在 moves 的前面，而不是后面
                    best_moves =   [[index_b, a]]+moves

    return max_score, best_moves
def recursive_Strategy(A, B):
     _,move=recursive_StrategyAndScore(A,B)
     return move

if __name__ == "__main__":
    A= [9, 2, 7, 13, 6, 10]
    B= [9, 110, 1300]
    # print(recursive(A,B))
    s,st=recursive_StrategyAndScore(A, B)
    print(s)
    print(st)
