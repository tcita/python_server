import multiprocessing
import random
import numpy as np
import time
import pickle
from multiprocessing import Pool
import scipy.linalg
import json
import os
import torch

from AI_algorithm.tool.tool import complete_best_moves, calculate_score_by_strategy


# ---------------------------
# 游戏规则函数
# ---------------------------

# 初始化一副牌（1到13，4种花色）
def init_deck():
    return [i for i in range(1, 14) for _ in range(4)]  # 创建一个包含四副牌的列表



def deal_cards(json_file="AI_algorithm/json/data_raw.json", seed=None):
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        random.seed(seed)

    # 检查文件是否存在
    full_path = os.path.join(os.path.dirname(__file__), "json", os.path.basename(json_file))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"文件未找到: {full_path}")

    try:
        # 从JSON文件读取数据
        with open(full_path, 'r') as f:
            cases = json.load(f)

        # 随机选择一个案例
        case = random.choice(cases)

        # 从案例中提取A和B的牌
        A = case.get('A', [])
        B = case.get('B', [])

        # 如果JSON中没有提供牌，则使用默认的随机发牌逻辑
        if not A or not B:
            raise ValueError("A或B为空")

        return A, B

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"读取JSON文件时出错: {e}")
        raise


# 洗牌并发牌，从JSON文件中读取A和B
# def deal_cards(json_file="AI_algorithm/json/transformer_error_cases.json"):
#     # 检查文件是否存在
#     if not os.path.exists(json_file):
#         # 如果文件不存在，尝试相对路径
#         json_file = "json/transformer_error_cases.json"
#         if not os.path.exists(json_file):
#
#             raise FileNotFoundError(f"文件未找到: {json_file}")
#
#     try:
#         # 从JSON文件读取数据
#         with open(json_file, 'r') as f:
#             cases = json.load(f)
#
#         # 随机选择一个案例
#         case = random.choice(cases)
#
#         # 提取A和B
#         A = case['A']
#         B = case['B']
#
#         print(f"从JSON加载案例: A={A}, B={B}")
#         return A, B
#
#     except Exception as e:
#
#         raise Exception(f"读取JSON文件时出错: {e}")


def get_best_insertion_score(A, card):
    max_score = -float('inf')
    best_pos = -1

    # 遍历所有可能的插入位置 (从位置0开始)
    for pos in range(0, len(A) + 1):
        score, _, _, _, _ = simulate_insertion(A, card, pos)

        # 找到最大的得分
        if score > max_score:
            max_score = score
            best_pos = pos

    return best_pos, max_score




# def choose_insertion(genome, A, B, x, remaining_B):
# x是不能构成匹配的牌组成的数组

# 像玩家一样做决策
def calculate_future_score(A, remaining_B):


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

        _, _, _, _, newA = simulate_insertion(A, card1, bestpos)
        # 把card2插入新的A
        _, score_card2 = get_best_insertion_score(newA, card2)

        # 计算它们的和

        best_score1 = score_card1 + score_card2

        # 先插入card2计算得分
        bestpos, score_card2 = get_best_insertion_score(A, card2)
        # 插入card2后的A变成了什么
        _, _, _, _, newA = simulate_insertion(A, card2, bestpos)
        # 把card1插入新的A
        _, score_card1 = get_best_insertion_score(newA, card1)

        # 计算它们的和

        best_score2 = score_card1 + score_card2

        return max(best_score2, best_score1)

    return future_score  # 默认返回0


def simulate_insertion(A, x, pos):
    candidate_A = A.copy()  # 复制A玩家的牌以便模拟插入

    # 处理空列表的情况
    if len(candidate_A) == 0:
        candidate_A.append(x)  # 如果A为空，则直接插入x
        return 0, 0, len(candidate_A), 0, candidate_A  # 返回相关信息

    # # 确保不在第一个元素之前插入
    # if pos == 0:
    #      raise ValueError("pos 不能为 0")  # 抛出异常，不允许在第一个位置之前插入

    # 插入x到pos位置
    candidate_A.insert(pos, x)  # 在指定位置插入x

    # 向左和向右搜索匹配的元素
    left_idx, right_idx = None, None  # 初始化左右索引为None

    # 向左搜索
    for i in range(pos - 1, -1, -1):  # 从pos-1开始向前搜索
        if candidate_A[i] == x:  # 如果找到匹配的元素
            left_idx = i  # 记录左侧匹配元素的位置
            break  # 结束循环

    # 向右搜索
    for j in range(pos + 1, len(candidate_A)):  # 从pos+1开始向后搜索
        if candidate_A[j] == x:  # 如果找到匹配的元素
            right_idx = j  # 记录右侧匹配元素的位置
            break  # 结束循环

    # 如果没有匹配元素
    if left_idx is None and right_idx is None:
        return 0, 0, len(candidate_A), 0, candidate_A  # 返回相关信息

    # 计算距离和区间的选择
    left_distance = pos - left_idx if left_idx is not None else float('inf')  # 计算左侧距离
    right_distance = right_idx - pos if right_idx is not None else float('inf')  # 计算右侧距离

    if left_distance < right_distance:
        start, end = min(pos, left_idx), max(pos, left_idx)  # 选择左侧匹配元素作为区间起点和终点
    elif right_distance < left_distance:
        start, end = min(pos, right_idx), max(pos, right_idx)  # 选择右侧匹配元素作为区间起点和终点
    else:
        left_sum = sum(candidate_A[min(pos, left_idx):max(pos, left_idx) + 1])  # 计算左侧区间的和
        right_sum = sum(candidate_A[min(pos, right_idx):max(pos, right_idx) + 1])  # 计算右侧区间的和
        if left_sum >= right_sum:
            start, end = min(pos, left_idx), max(pos, left_idx)  # 选择左侧区间
        else:
            start, end = min(pos, right_idx), max(pos, right_idx)  # 选择右侧区间

    # 计算区间和
    removal_interval = candidate_A[start:end + 1]  # 获取要移除的区间
    score = sum(removal_interval)  # 计算区间和

    # 移除区间内的所有元素
    new_A = candidate_A[:start] + candidate_A[end + 1:]  # 生成新的牌堆

    return score, len(removal_interval), len(new_A), 1, new_A  # 返回相关信息


# 选择最优插入位置的策略
# 参数解释  x  B中的特定元素  x将使用遍历B来获取  如 foreach(x in B)
# remaining_B  还未被插入得分的B中的元素
# def genome_choose_insertion(genome, A, x, remaining_B):
#     best_value, best_move = -float('inf'), None  # 初始化最佳评估值和最佳移动
#
#     # 计算新增特征
#     sum_A = sum(A)  # A的元素总和
#
#     # 计算B的元素总和 (当前x + 剩余的B)
#     B_elements = [x] + remaining_B
#
#     # 计算B与A的交集的数量
#     intersection_count = len(set(B_elements) & set(A))
#
#
#     possible_moves = []  # 存储所有候选插入位置的得分
#
#     # 从位置0开始插入  现在可以从0开始了
#     for pos in range(0, len(A) + 1):  # 尝试所有可能的插入位置
#         score, removal_length, new_length, match_found, new_A = simulate_insertion(A, x, pos)  # 模拟插入并获取结果
#         current_score = score  # 当前得分
#
#         future_score = calculate_future_score(new_A, remaining_B)  # 计算未来的得分
#
#         # 修改为6个特征向量
#         features = np.array([
#             current_score,       # 当前得分 (就是移除的子列表的总分)
#             removal_length,      # 移除长度
#             new_length,          # 新长度
#             future_score,        # 未来得分
#             sum_A,               # A的元素总和
#             intersection_count,  # B与A的交集数量
#         ], dtype=float)  # 特征向量
#         # 使用CUDA加速计算评估值（如果可用）
#
#         value = np.dot(genome, features)  # 基因组的评估值
#
#
#         possible_moves.append((value, pos, current_score, new_A))  # 将当前插入位置及其得分保存到列表中
#
#     # 如果对于特定的B元素 有多个可能的插入位置，选择得分最高的那个插入位置
#     if possible_moves:
#         best_move = max(possible_moves, key=lambda move: move[0])
#
#     #  处理找不到插入位置的情况
#     if best_move is None:
#         # possible_moves = []
#         # for pos in range(1, len(A) + 1):  # 尝试所有可能的插入位置
#         #     score, removal_length, new_length, match_found, new_A = simulate_insertion(A, x, pos)  # 模拟插入并获取结果
#         #     possible_moves.append((score, pos, new_A))  # 将当前插入位置及其得分保存到列表中
#
#         # 默认插入到末尾，否则没有返回值
#         print("No suitable insertion position found.")
#         return len(A), 0, A
#
#         # best_move = max(possible_moves, key=lambda move: move[0])
#
#     pos, score, new_A = best_move[1], best_move[2], best_move[3]  # 提取最佳插入位置、得分和新牌堆
#     return pos, score, new_A  # 返回最佳插入位置、得分和新牌堆


# # 模拟一轮游戏
# def simulate_round(genome):
#     A, B = deal_cards()  # 发牌
#     round_score = 0  # 初始化本轮得分
#     for i, x in enumerate(B):  # 遍历B玩家的每一张牌
#         remaining_B = B[i + 1:]  # 获取后续未处理的B牌
#         pos, score, A = genome_choose_insertion(genome, A, x, remaining_B)  # 选择最优插入位置并更新牌堆
#         round_score += score  # 累加得分
#     return round_score  # 返回本轮总得分


# 评估基因组的适应度
def evaluate_all_insertion_by_genome(genome, A, B):
    """
    使用基因组评估所有可能的B牌处理顺序，返回最高得分和相应策略

    参数：
    - genome: 基因组权重
    - A: A玩家的牌
    - B: B玩家的牌

    返回：
    - best_score: 最高得分
    - best_strategy: 最佳策略
    """
    import torch
    import numpy as np

    # 检查CUDA可用性并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    # 将基因组转换为torch张量以便GPU加速
    genome_tensor = torch.tensor(genome, dtype=torch.float32, device=device)

    # 使用PyTorch进行特征计算以实现GPU加速
    def compute_card_values(A_copy, B, genome_tensor):
        # 存储每张牌的评估值
        card_values = []
        for i, card in enumerate(B):
            # 对每张牌，计算其在不同位置插入的最大评估值
            remaining_B = [B[j] for j in range(len(B)) if j != i]
            best_value = -float('inf')
            best_pos = -1
            best_score = 0
            best_new_A = None

            # 并行计算所有可能的插入位置的特征
            for pos in range(len(A_copy) + 1):
                # 模拟插入并计算得分
                score, removal_length, new_length, match_found, new_A = simulate_insertion(A_copy, card, pos)
                future_score = calculate_future_score(new_A, remaining_B)

                # 使用PyTorch计算特征（适用于GPU加速）
                sum_A = sum(A_copy)
                intersection_count = len(set([card] + remaining_B) & set(A_copy))

                # 构建特征向量
                features = torch.tensor([
                    score,  # 当前得分
                    removal_length,  # 移除长度
                    new_length,  # 新长度
                    future_score,  # 未来得分
                    sum_A,  # A的元素总和
                    intersection_count,  # B与A的交集数量
                ], dtype=torch.float32, device=device)

                # 在GPU上计算点积
                value = torch.dot(genome_tensor, features).item()

                # 更新最佳值
                if value > best_value:
                    best_value = value
                    best_pos = pos
                    best_score = score
                    best_new_A = new_A

            # 记录每张牌的最佳插入信息
            card_values.append((i, best_value, best_pos, best_score, best_new_A))

        return card_values

    # 计算卡值
    card_values = compute_card_values(A.copy(), B, genome_tensor)

    # 根据评估值对卡进行排序
    card_values.sort(key=lambda x: x[1], reverse=True)

    # 根据排序后的顺序生成策略
    strategy = [(card_idx, pos) for card_idx, _, pos, _, _ in card_values]




    return strategy


def evaluate_genome(genome, num_rounds=1000, seed_base=42):
    import torch
    import numpy as np
    from AI_algorithm.tool.tool import calculate_score_by_strategy

    # 检查并设置 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"开始评估基因组，总共 {num_rounds} 轮...")

    # 将基因组转换为 GPU 张量
    genome_tensor = torch.tensor(genome, dtype=torch.float32, device=device)

    # 批处理大小，根据GPU内存调整
    batch_size = 64 if device.type == 'cuda' else 32

    # 预分配结果数组
    total_scores = torch.zeros(num_rounds, dtype=torch.float32, device=device)

    # 添加进度输出
    progress_interval = max(1, num_rounds // 10)  # 每10%输出一次

    # 批量处理评估任务
    for batch_start in range(0, num_rounds, batch_size):
        batch_end = min(batch_start + batch_size, num_rounds)
        batch_size_actual = batch_end - batch_start

        # 为当前批次生成所有A和B
        batch_A = []
        batch_B = []
        for _ in range(batch_size_actual):
            A, B = deal_cards(seed=seed_base + batch_start + _)
            batch_A.append(A)
            batch_B.append(B)

        # 并行评估批次使用GPU
        batch_scores = torch.zeros(batch_size_actual, dtype=torch.float32, device=device)

        for j in range(batch_size_actual):
            strategy = evaluate_all_insertion_by_genome(genome, batch_A[j], batch_B[j])
            batch_scores[j] = calculate_score_by_strategy(batch_A[j], batch_B[j], strategy)

        # 复制批次得分到总得分
        total_scores[batch_start:batch_end] = batch_scores

        # 进度跟踪
        if (batch_start + batch_size) % progress_interval == 0:
            print(f"进度: {(batch_start + batch_size) / num_rounds * 100:.2f}%")

    # 计算最终平均分
    mean_score = torch.mean(total_scores).item()

    print(f"基因组评估完成:")
    print(f"平均得分: {mean_score}")

    return mean_score  # 返回平均得分
def evaluate_genomes_return_fitness(population, num_rounds=1000):
    """
    使用多进程评估基因组适应度，添加超时和异常处理

    参数：
    - population: 种群
    - num_rounds: 评估轮数

    - timeout: 超时时间（秒）

    返回：
    - fitnesses: 适应度列表
    """
    # print(f"使用 {num_processes} 个进程评估 {len(population)} 个基因组")

    fitnesses = []

    # 调试模式：同步调用
    for genome in population:
        try:
            print(f"正在评估第 {population.index(genome) + 1} 个基因组")
            fitness = evaluate_genome(genome, num_rounds)
            print(f"基因组 {genome} 评估完成，适应度: {fitness}")
            fitnesses.append(fitness)
        except Exception as e:
            print(f"基因组 {genome} 评估发生异常: {e}")
            fitnesses.append(-float('inf'))

    return fitnesses


# 岛屿模型实现，用于增加种群多样性
def island_model_evolution(population, fitnesses, pop_size, tournament_size, mutation_strength,
                           num_rounds, islands=4, migration_interval=10, migration_rate=0.1,
                           generation=0, max_generations=60):
    """
    实现岛屿模型进化，将总人口分成几个独立'岛屿'，定期交换个体

    参数：
    - population: 当前种群
    - fitnesses: 当前种群的适应度值
    - pop_size: 总人口规模
    - tournament_size: 锦标赛选择规模
    - mutation_strength: 变异强度
    - islands: 岛屿数量
    - migration_interval: 多少代进行一次迁移
    - migration_rate: 每次迁移的个体比例
    - generation: 当前代数
    - max_generations: 最大代数

    返回：
    - new_population: 进化后的新种群
    - new_fitnesses: 新种群的适应度
    """
    # 动态调整迁移率 - 随着进化过程增加交流
    progress_ratio = generation / max_generations if max_generations > 0 else 0.5
    adaptive_migration_rate = migration_rate * (1.0 + progress_ratio)  # 迁移率逐渐增加到原来的2倍

    # 计算每个岛屿的规模和迁移数量
    island_size = pop_size // islands
    migration_size = int(island_size * adaptive_migration_rate)

    # 将总人口分割成岛屿
    island_populations = []
    island_fitnesses = []

    for i in range(islands):
        start_idx = i * island_size
        end_idx = start_idx + island_size if i < islands - 1 else pop_size
        island_populations.append(population[start_idx:end_idx])
        island_fitnesses.append(fitnesses[start_idx:end_idx])

    # 每个岛屿独立进化
    new_island_populations = []
    new_island_fitnesses = []

    for i in range(islands):
        # 岛内进化
        # 选择精英
        sorted_indices = sorted(range(len(island_fitnesses[i])),
                              key=lambda k: island_fitnesses[i][k], reverse=True)
        elitism_count = int(0.1 * len(island_populations[i]))
        elites = [island_populations[i][idx] for idx in sorted_indices[:elitism_count]]

        # 锦标赛选择
        selected = []
        for _ in range(len(island_populations[i]) - elitism_count):
            candidates = random.sample(range(len(island_populations[i])), tournament_size)
            winner_idx = max(candidates, key=lambda idx: island_fitnesses[i][idx])
            selected.append(island_populations[i][winner_idx])

        # 交叉和变异
        next_population = elites.copy()
        while len(next_population) < len(island_populations[i]):
            parent1, parent2 = random.sample(selected, 2)
            child = [(p1 + p2) / 2 if random.random() < 0.7 else p1
                    for p1, p2 in zip(parent1, parent2)]
            # 修复变异部分的错误
            mutation_strength = 0.7  # 增加变异强度
            child = [gene + random.gauss(0, mutation_strength) for gene in child]
            next_population.append(child)

        new_island_populations.append(next_population)

    # 评估新岛屿种群适应度
    new_island_fitnesses = [evaluate_genomes_return_fitness(pop, num_rounds)
                            for pop in new_island_populations]

    # 迁移过程 (如果当前代数是迁移间隔的倍数)
    # 注意：在实际使用时，需要传入当前代数作为参数，这里假设每次调用都执行迁移
    for i in range(islands):
        # 选择当前岛屿的最佳个体进行迁移
        sorted_indices = sorted(range(len(new_island_fitnesses[i])),
                              key=lambda k: new_island_fitnesses[i][k], reverse=True)
        migrants_indices = sorted_indices[:migration_size]
        migrants = [new_island_populations[i][idx] for idx in migrants_indices]

        # 将个体迁移到下一个岛屿(环形拓扑)
        target_island = (i + 1) % islands

        # 在目标岛屿中，替换最差的个体
        target_sorted_indices = sorted(range(len(new_island_fitnesses[target_island])),
                                     key=lambda k: new_island_fitnesses[target_island][k])
        for j, migrant in enumerate(migrants):
            if j < len(target_sorted_indices):
                replace_idx = target_sorted_indices[j]
                new_island_populations[target_island][replace_idx] = migrant

    # 重新评估迁移后的适应度
    new_island_fitnesses = [evaluate_genomes_return_fitness(pop, num_rounds)
                            for pop in new_island_populations]

    # 合并所有岛屿种群
    new_population = []
    new_fitnesses = []
    for pop, fit in zip(new_island_populations, new_island_fitnesses):
        new_population.extend(pop)
        new_fitnesses.extend(fit)

    # 如果合并后的种群大小超过了原始种群大小，截断到原始大小
    if len(new_population) > pop_size:
        combined = list(zip(new_population, new_fitnesses))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        new_population = [x[0] for x in sorted_combined[:pop_size]]
        new_fitnesses = [x[1] for x in sorted_combined[:pop_size]]

    return new_population, new_fitnesses


# 差分进化算法实现
def differential_evolution(population, fitnesses, pop_size, F=0.8, CR=0.5, num_rounds=1000,
                           generation=0, max_generations=60):
    """
    实现差分进化算法

    参数：
    - population: 当前种群
    - fitnesses: 当前种群的适应度值
    - pop_size: 种群规模
    - F: 缩放因子(典型值:0.5-1.0)
    - CR: 交叉概率(典型值:0.1-0.9)
    - num_rounds: 评估轮数

    - generation: 当前代数
    - max_generations: 最大代数

    返回：
    - new_population: 进化后的新种群
    - new_fitnesses: 新种群的适应度
    """
    # 动态调整F和CR参数
    progress_ratio = generation / max_generations if max_generations > 0 else 0.5
    adaptive_F = F * (1.0 - 0.3 * progress_ratio)  # F从初始值逐渐降低30%
    adaptive_CR = min(0.9, CR + 0.3 * progress_ratio)  # CR从初始值逐渐增加，最大到0.9

    new_population = []

    # 为每个个体进行差分进化
    for i in range(pop_size):
        target = population[i]

        # 随机选择三个不同的个体，且与当前个体不同
        candidates = list(range(pop_size))
        candidates.remove(i)
        a, b, c = random.sample(candidates, 3)

        # 选择的三个个体
        x_a = population[a]
        x_b = population[b]
        x_c = population[c]

        # 生成突变向量，使用动态调整的F
        mutant = [x_a[j] + adaptive_F * (x_b[j] - x_c[j]) for j in range(len(target))]

        # 交叉操作，使用动态调整的CR
        trial = []
        for j in range(len(target)):
            if random.random() < adaptive_CR or j == random.randint(0, len(target)-1):
                trial.append(mutant[j])
            else:
                trial.append(target[j])

        new_population.append(trial)

    # 评估新种群
    new_fitnesses = evaluate_genomes_return_fitness(new_population, num_rounds)

    # 选择操作：如果新个体更好，则替换旧个体
    for i in range(pop_size):
        if fitnesses[i] > new_fitnesses[i]:
            new_population[i] = population[i]
            new_fitnesses[i] = fitnesses[i]

    return new_population, new_fitnesses


# 协方差矩阵自适应(CMA-ES)算法实现
def cmaes_evolve(population, fitnesses, pop_size, num_rounds=1000,  generation=0, max_generations=60):
    """
    实现协方差矩阵自适应进化策略(CMA-ES)

    参数：
    - population: 当前种群
    - fitnesses: 当前种群的适应度值
    - pop_size: 种群规模
    - num_rounds: 评估轮数

    - generation: 当前代数
    - max_generations: 最大代数

    返回：
    - new_population: 进化后的新种群
    - new_fitnesses: 新种群的适应度
    """
    # 计算当前种群的均值向量
    n = len(population[0])  # 个体维度
    mean = np.zeros(n)
    for ind in population:
        mean += np.array(ind)
    mean /= pop_size

    # 动态调整步长sigma，随着进化过程逐渐减小
    progress_ratio = generation / max_generations if max_generations > 0 else 0.5
    sigma = max(0.1, 0.8 - 0.6 * progress_ratio)  # 从0.8逐渐降低到0.2

    # 初始化协方差矩阵为单位矩阵
    C = np.identity(n)

    # 计算种群中个体与均值的差异，构建协方差矩阵
    if pop_size > 1:  # 确保有足够样本计算协方差
        diff_matrix = np.array([np.array(ind) - mean for ind in population])
        C = np.cov(diff_matrix.T) + 1e-8 * np.identity(n)  # 添加小值避免奇异矩阵

    # 生成新种群
    new_population = []

    # 特征值分解
    try:
        eigvals, eigvecs = scipy.linalg.eigh(C)
        # 确保特征值为正
        eigvals = np.maximum(eigvals, 1e-8)

        # 生成新的个体
        for _ in range(pop_size):
            # 生成标准正态分布的随机向量
            z = np.random.randn(n)
            # 变换为多元正态分布
            s = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(z)
            # 生成新个体
            new_ind = list(mean + sigma * s)
            new_population.append(new_ind)
    except:
        # 如果特征值分解失败，使用简单的随机变异
        for _ in range(pop_size):
            new_ind = list(mean + sigma * np.random.randn(n))
            new_population.append(new_ind)

    # 评估新种群
    new_fitnesses = evaluate_genomes_return_fitness(new_population, num_rounds)

    return new_population, new_fitnesses


# 遗传算法过程
def genetic_algorithm(pop_size=40, generations=60, num_rounds=16, elitism_ratio=0.1, tournament_size=3,
                      evolution_methods=['standard', 'island', 'de', 'cmaes'],
                      method_probs=[0.30, 0.40, 0.15, 0.15] , early_stop_generations=7, early_stop_threshold=0.01):
    """
    遗传算法主函数

    参数：
    - pop_size: 种群大小
    - generations: 最大进化代数
    - num_rounds: 评估每个基因组的回合数
    - elitism_ratio: 精英比例
    - tournament_size: 锦标赛选择规模

    - evolution_methods: 可用的进化方法列表
    - method_probs: 各进化方法的使用概率
    - early_stop_generations: 早停的连续代数
    - early_stop_threshold: 早停的改善阈值

    返回：
    - best_genome: 最佳基因组
    """
    start_time = time.time()  # 记录开始时间
    # 在genetic_algorithm函数中修改初始化种群的代码
    population = []
    for _ in range(pop_size):
        # 为当前得分和未来得分特征赋予更高的初始权重
        genome = []
        for i in range(6):
            if i == 0 or i == 3:  # 当前得分和未来得分的索引
                genome.append(random.uniform(0, 2))  # 更高的正向权重
            else:
                genome.append(random.uniform(-1, 1))
        population.append(genome)
    best_fitness_history, avg_fitness_history = [], []  # 初始化历史最佳适应度和平均适应度列表
    best_genome, best_fitness = None, -float('inf')  # 初始化最佳基因组和最佳适应度
    elitism_count = int(elitism_ratio * pop_size)  # 计算精英数量
    method_history = []  # 记录每代使用的进化方法

    # 早停相关变量
    early_stop_counter = 0
    last_best_fitness = -float('inf')

    for gen in range(generations):  # 迭代每个世代
        fitnesses = evaluate_genomes_return_fitness(population, num_rounds)  # 评估种群适应度
        gen_best, gen_avg = max(fitnesses), np.mean(fitnesses)  # 获取当前世代的最佳适应度和平均适应度
        best_fitness_history.append(gen_best)  # 记录最佳适应度
        avg_fitness_history.append(gen_avg)  # 记录平均适应度

        # 早停检查
        improvement = gen_best - last_best_fitness

        # 修复除法错误
        if last_best_fitness != -float('inf') and abs(last_best_fitness) > 1e-10:
            relative_improvement = improvement / abs(last_best_fitness)
        else:
            # 第一代或last_best_fitness接近0或无穷时，使用绝对改善判断
            relative_improvement = 1.0 if improvement > early_stop_threshold else 0.0

        if relative_improvement <= early_stop_threshold:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_generations:
                print(f"早停触发: {early_stop_generations}代内适应度改善低于{early_stop_threshold:.2%}")
                break
        else:
            early_stop_counter = 0  # 重置计数器

        last_best_fitness = gen_best

        if gen_best > best_fitness:  # 更新最佳基因组和适应度
            best_fitness, best_genome = gen_best, population[fitnesses.index(gen_best)]

        print(f"Generation {gen + 1}: Best Fitness = {gen_best:.2f}, Average Fitness = {gen_avg:.2f}")  # 打印当前世代信息

        sorted_population = [x for _, x in
                             sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]  # 按适应度排序种群
        elites = sorted_population[:elitism_count]  # 选择精英

        selected = [population[max(random.sample(range(pop_size), tournament_size), key=lambda i: fitnesses[i])] for _
                    in range(pop_size - elitism_count)]  # 通过锦标赛选择剩余的基因组

        next_population = elites.copy()  # 初始化下一代种群
        while len(next_population) < pop_size:  # 生成下一代
            parent1, parent2 = random.sample(selected, 2)  # 选择两个父本
            child = [(p1 + p2) / 2 if random.random() < 0.7 else p1 for p1, p2 in zip(parent1, parent2)]  # 交叉产生子代
            mutation_rate = max(0.1, 0.3 - (gen / generations) * 0.2)  # 计算变异率
            child = [gene + random.gauss(0, 0.5) if random.random() < mutation_rate else gene for gene in child]  # 变异
            next_population.append(child)  # 添加子代到下一代种群

        # 根据概率选择进化方法
        method_idx = np.random.choice(len(evolution_methods), p=method_probs)
        method = evolution_methods[method_idx]
        method_history.append(method)

        if method == 'standard':
            pass  # 使用标准遗传算法
        elif method == 'island':
            next_population, _ = island_model_evolution(next_population, fitnesses, pop_size, tournament_size, 0.5, num_rounds, generation=gen, max_generations=generations)
        elif method == 'de':
            next_population, _ = differential_evolution(next_population, fitnesses, pop_size, F=0.8, CR=0.5, num_rounds=num_rounds, generation=gen, max_generations=generations)
        elif method == 'cmaes':
            next_population, _ = cmaes_evolve(next_population, fitnesses, pop_size, num_rounds=num_rounds, generation=gen, max_generations=generations)

        population = next_population  # 更新种群

    end_time = time.time()  # 记录结束时间
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")  # 打印总执行时间
    print(f"Completed Generations: {gen + 1} of {generations}")  # 打印实际完成的代数

    # 分析不同进化方法的性能
    analyze_evolution_methods(best_fitness_history, method_history)

    return best_genome  # 返回最佳基因组


def save_best_genome(genome, filename="trained/best_genome.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(genome, file)  # 保存最佳基因组到文件
    print(f"Best genome saved to {filename}")  # 打印保存信息

    # 打印基因组各特征的权重
    feature_names = [
        "当前得分", "移除长度", "新长度",  "未来得分",
        "A元素总和",  "B与A交集数量"
    ]

    print("\n基因组特征权重:")
    for i, (name, weight) in enumerate(zip(feature_names, genome)):
        print(f"{name}: {weight:.4f}")


# 分析进化方法的性能
def analyze_evolution_methods(best_fitness_history, method_history):
    """
    分析不同进化方法的性能

    参数：
    - best_fitness_history: 每代的最佳适应度历史记录
    - method_history: 每代使用的进化方法历史记录
    """
    import matplotlib.pyplot as plt

    # 按照进化方法分组
    method_performance = {}
    for method, fitness in zip(method_history, best_fitness_history):
        if method not in method_performance:
            method_performance[method] = []
        method_performance[method].append(fitness)

    # 计算每种方法的平均性能和最大性能
    print("\n各进化方法性能分析:")
    for method, fitnesses in method_performance.items():
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        improve_rate = (fitnesses[-1] - fitnesses[0]) / fitnesses[0] if len(fitnesses) > 1 else 0
        print(f"{method}方法: 平均适应度={avg_fitness:.2f}, 最大适应度={max_fitness:.2f}, 改进率={improve_rate:.2%}")



# def GA(genome, A, B):
#     """
#     尝试所有可能的B牌处理顺序，返回最高得分
#     """
#     import itertools
#
#     # 获取所有可能的B牌处理顺序
#     all_orders = list(itertools.permutations(range(len(B))))
#
#     best_score = -float('inf')
#     best_A = None
#
#     # 尝试每种可能的顺序
#     for order in all_orders:
#         A_copy = A.copy()
#         current_score = 0
#
#         # 按照当前顺序处理B牌
#         for idx in order:
#             x = B[idx]
#             # 计算剩余未处理的B牌
#             remaining_indices = [i for i in order if i > order.index(idx)]
#             remaining_B = [B[i] for i in remaining_indices]
#
#             # 选择最优插入位置
#             pos, score, A_copy = genome_choose_insertion(genome, A_copy, x, remaining_B)
#             current_score += score
#
#         # 如果当前顺序得分更高，更新最佳得分
#         if current_score > best_score:
#             best_score = current_score
#             best_A = A_copy
#
#     return best_score  # 返回最高得分

def load_best_genome(filename="../trained/best_genome.pkl"):
    try:
        # 尝试打开并加载文件
        with open(filename, 'rb') as file:
            genome = pickle.load(file)
        # print(f"genome loaded from {filename}")
        return genome
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



if __name__ == "__main__":
    genome = genetic_algorithm()  # 运行遗传算法获取最佳基因组
    print("\nGenome model : ", genome)  # 打印最佳基因组
    # evaluate_final_model(genome)  # 评估最终模型性能
    save_best_genome(genome)  # 保存最佳基因组
    #
    # genome=load_best_genome()
    # GA_partial = partial(GA, genome)
    # modsummery(GA_partial,2000)
