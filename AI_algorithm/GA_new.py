import random
import numpy as np
import time
import pickle
from multiprocessing import Pool
import os  # 用于创建目录


# ---------------------------
# 游戏规则与模拟函数
# ---------------------------

# 初始化一副牌（1到13，4种花色）
def init_deck():
    """创建一副标准的52张扑克牌（点数1-13，每种点数4张）"""
    return [i for i in range(1, 14) for _ in range(4)]


# 洗牌并发牌，A玩家6张（唯一），B玩家3张
def deal_cards():
    """
    洗牌并分发手牌。
    A获得6张不同的牌。
    B获得3张牌。
    确保A中的牌是唯一的。
    """
    deck = init_deck()
    random.shuffle(deck)

    A = []
    a_unique_checker = set()  # 用于快速检查A中是否已有某张牌
    deck_idx = 0  # 当前从牌堆抽到的索引

    # 抽取A的牌，确保唯一性
    while len(A) < 6 and deck_idx < len(deck):
        card = deck[deck_idx]
        if card not in a_unique_checker:
            A.append(card)
            a_unique_checker.add(card)
        deck_idx += 1

    # 健壮性检查：如果牌堆不足以抽出6张唯一的牌（理论上不可能发生）
    if len(A) < 6:
        raise Exception(f"无法为A抽取6张唯一的牌, 只抽到了 {len(A)} 张。")

    # 从牌堆剩余部分抽取B的牌
    remaining_deck = deck[deck_idx:]
    if len(remaining_deck) < 3:
        raise Exception(f"牌堆剩余牌数不足 ({len(remaining_deck)}) 为B抽取3张牌。")

    B = [remaining_deck.pop(random.randrange(len(remaining_deck))) for _ in range(3)]  # 从剩余牌中随机抽取3张给B

    return A, B


# 模拟将牌x插入A的pos位置，并计算得分和更新A
def simulate_insertion(A, x, pos):
    """
    模拟将牌 x 插入到列表 A 的指定位置 pos (0-based index)。
    计算得分，更新列表 A。

    Args:
        A (list): 玩家A当前的手牌。
        x (int): 要插入的牌。
        pos (int): 插入的位置索引 (0 <= pos <= len(A))。

    Returns:
        tuple: (得分, 移除的牌数, 新A的长度, 是否找到匹配(1或0), 新的A列表)
    """
    candidate_A = A.copy()

    # 插入 x 到指定位置
    # pos=0 表示插入到最前面, pos=len(A) 表示插入到最后面
    candidate_A.insert(pos, x)

    left_idx, right_idx = None, None

    # 向左搜索匹配的牌 x
    for i in range(pos - 1, -1, -1):
        if candidate_A[i] == x:
            left_idx = i
            break

    # 向右搜索匹配的牌 x
    for j in range(pos + 1, len(candidate_A)):
        if candidate_A[j] == x:
            right_idx = j
            break

    # 如果没有找到匹配
    if left_idx is None and right_idx is None:
        return 0, 0, len(candidate_A), 0, candidate_A

    # 选择匹配区间
    start, end = -1, -1
    if left_idx is not None and right_idx is not None:
        # 如果左右都有匹配，选择距离更近的
        # 如果距离相等，选择区间和更大的
        left_distance = pos - left_idx
        right_distance = right_idx - pos
        if left_distance < right_distance:
            start, end = left_idx, pos
        elif right_distance < left_distance:
            start, end = pos, right_idx
        else:
            left_sum = sum(candidate_A[left_idx: pos + 1])
            right_sum = sum(candidate_A[pos: right_idx + 1])
            if left_sum >= right_sum:
                start, end = left_idx, pos
            else:
                start, end = pos, right_idx
    elif left_idx is not None:  # 只有左匹配
        start, end = left_idx, pos
    else:  # 只有右匹配 (right_idx is not None)
        start, end = pos, right_idx

    # 计算得分并移除区间
    removal_interval = candidate_A[start: end + 1]
    score = sum(removal_interval)
    new_A = candidate_A[:start] + candidate_A[end + 1:]

    return score, len(removal_interval), len(new_A), 1, new_A


# ---------------------------
# 遗传算法相关函数
# ---------------------------

# 根据基因组选择最优插入位置 (简化版)
def genome_choose_insertion(genome, A, x):
    """
    使用基因组评估并选择牌 x 插入 A 的最佳位置。
    评估基于当前步骤的直接结果。

    Args:
        genome (list): 遗传算法的基因组（权重列表，长度为4）。
        A (list): 玩家A当前的手牌。
        x (int): 要插入的牌。

    Returns:
        tuple: (最佳插入位置, 该位置得分, 更新后的A列表)
    """
    best_value = -float('inf')
    best_move = None  # (评估值, 位置, 得分, 新A列表)
    possible_moves = []

    # 遍历所有可能的插入位置 (从0到len(A))
    for pos in range(len(A) + 1):
        # 模拟插入，获取结果
        score, removal_length, new_length, match_found, new_A = simulate_insertion(A, x, pos)

        # 构建特征向量 (4个特征)
        # 特征: [当前得分, 移除牌数, 新手牌长度, 是否找到匹配]
        features = np.array([score, removal_length, new_length, match_found], dtype=float)

        # 使用基因组（权重）计算该移动的评估值
        value = np.dot(genome, features)

        possible_moves.append((value, pos, score, new_A))

    # 如果没有可能的移动（只可能发生在初始A为空时，但simulate_insertion已处理）
    # 按理说 possible_moves 不会为空
    if not possible_moves:
        # 作为备用方案，如果 A 为空，则默认插入在位置0
        # 但 simulate_insertion(A=[], x, 0) 会返回有效结果，所以这里理论上不会执行
        print(f"警告: A={A}, x={x} 时没有找到可能的移动?")
        score, _, _, _, new_A = simulate_insertion(A, x, 0)
        return 0, score, new_A  # 默认返回插入到位置0

    # 选择评估值最高的移动
    best_move = max(possible_moves, key=lambda move: move[0])

    # 提取最佳移动的信息
    _, best_pos, best_score, best_new_A = best_move
    return best_pos, best_score, best_new_A


# 模拟一轮完整的游戏 (处理B中的所有牌)
def simulate_round(genome):
    """
    模拟一轮完整的游戏。
    发牌 -> 随机打乱B -> 依次处理B中的牌 -> 计算总分。

    Args:
        genome (list): 用于决策的基因组。

    Returns:
        int: 这一轮的总得分。
    """
    try:
        A_initial, B_initial = deal_cards()
    except Exception as e:
        print(f"发牌时出错: {e}")
        return 0  # 返回0分或者其他错误处理

    A_current = A_initial.copy()
    B_shuffled = B_initial.copy()
    random.shuffle(B_shuffled)  # *** 关键：随机打乱B的处理顺序 ***

    round_score = 0
    # 按随机顺序处理B中的每张牌
    for card_x in B_shuffled:
        # 使用基因组选择最佳插入位置并更新A
        pos, score, A_current = genome_choose_insertion(genome, A_current, card_x)
        round_score += score  # 累加得分

    return round_score


# 评估单个基因组的适应度
def evaluate_genome(genome, num_rounds=1000):
    """
    通过多次模拟游戏来评估一个基因组的适应度。

    Args:
        genome (list): 要评估的基因组。
        num_rounds (int): 模拟的游戏轮数。

    Returns:
        float: 该基因组的平均得分（适应度）。
    """
    scores = [simulate_round(genome) for _ in range(num_rounds)]
    # 使用平均分作为适应度，更标准
    fitness = np.mean(scores)
    # 原来的方法：fitness = np.median(scores) * 0.7 + np.mean(scores) * 0.3
    return fitness


# 使用多进程并行评估种群中所有基因组的适应度
def evaluate_genomes_with_processes(population, num_rounds=1000, num_processes=8):
    """使用多进程并行计算种群中每个基因组的适应度。"""
    # 确保进程数合理
    import multiprocessing
    max_proc = multiprocessing.cpu_count()
    num_processes = min(num_processes, max_proc)
    print(f"使用 {num_processes} 个进程进行评估...")

    with Pool(processes=num_processes) as pool:
        # starmap 用于传递多个参数给 evaluate_genome
        fitnesses = pool.starmap(evaluate_genome, [(genome, num_rounds) for genome in population])
    return fitnesses


# 遗传算法主过程
def genetic_algorithm(pop_size=1000, generations=60, num_rounds=1000, elitism_ratio=0.1, tournament_size=3,
                      num_processes=8):
    """
    执行遗传算法来优化游戏策略基因组。

    Args:
        pop_size (int): 种群大小。
        generations (int): 迭代的代数。
        num_rounds (int): 每代评估基因组时模拟的游戏轮数。
        elitism_ratio (float): 精英选择比例。
        tournament_size (int): 锦标赛选择的大小。
        num_processes (int): 用于并行评估的进程数。

    Returns:
        list: 找到的最佳基因组。
    """
    start_time = time.time()

    # 初始化种群，每个基因组是长度为4的随机权重列表 (-1 到 1)
    population = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(pop_size)]

    best_fitness_history = []
    avg_fitness_history = []
    best_genome_overall = None
    best_fitness_overall = -float('inf')

    elitism_count = int(elitism_ratio * pop_size)  # 精英数量

    for gen in range(generations):
        print(f"\n--- 第 {gen + 1}/{generations} 代 ---")

        # 1. 评估当前种群所有个体的适应度
        gen_start_time = time.time()
        fitnesses = evaluate_genomes_with_processes(population, num_rounds, num_processes)
        gen_eval_time = time.time() - gen_start_time
        print(f"评估耗时: {gen_eval_time:.2f} 秒")

        # 记录当前代的最佳和平均适应度
        gen_best_fitness = max(fitnesses)
        gen_avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        # 更新全局最佳基因组
        current_best_idx = fitnesses.index(gen_best_fitness)
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_genome_overall = population[current_best_idx]
            print(f"*** 新的全局最佳适应度: {best_fitness_overall:.4f} ***")

        print(f"当前代最佳适应度: {gen_best_fitness:.4f}")
        print(f"当前代平均适应度: {gen_avg_fitness:.4f}")
        print(f"当前代最佳基因组: {population[current_best_idx]}")

        # 2. 选择 (Selection)
        # 按适应度排序，方便选出精英
        sorted_indices = sorted(range(pop_size), key=lambda k: fitnesses[k], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]

        # 精英主义：直接保留适应度最高的个体
        elites = sorted_population[:elitism_count]

        # 锦标赛选择：为下一代选择父母
        selected_parents = []
        for _ in range(pop_size - elitism_count):  # 需要产生这么多后代
            # 随机选出 tournament_size 个个体进行比较
            tournament_contenders_indices = random.sample(range(pop_size), tournament_size)
            # 选出其中适应度最高的作为父母
            winner_idx = max(tournament_contenders_indices, key=lambda i: fitnesses[i])
            selected_parents.append(population[winner_idx])

        # 3. 交叉 (Crossover)
        next_population = elites.copy()  # 下一代从精英开始
        while len(next_population) < pop_size:
            # 从选出的父母中随机选两个
            parent1, parent2 = random.sample(selected_parents, 2)
            child = []
            # 对基因组的每个位置进行处理
            for i in range(4):  # 基因组长度为4
                # 70%概率取父母平均值（混合），30%概率直接继承parent1 (也可改为继承parent2或随机选一)
                if random.random() < 0.7:
                    child.append((parent1[i] + parent2[i]) / 2.0)
                else:
                    child.append(parent1[i])
            next_population.append(child)

        # 4. 变异 (Mutation)
        # 变异率随代数增加而降低 (可选策略)
        mutation_rate = max(0.05, 0.2 - (gen / generations) * 0.15)
        print(f"当前代变异率: {mutation_rate:.3f}")

        for i in range(elitism_count, pop_size):  # 不对精英进行变异
            genome = next_population[i]
            for j in range(4):  # 遍历基因组每个基因
                if random.random() < mutation_rate:
                    # 添加小的随机扰动（高斯分布）
                    genome[j] += random.gauss(0, 0.3)  # 标准差可以调整
                    # 可选：限制基因值范围，例如 [-1, 1] 或 [-2, 2]
                    genome[j] = max(-2.0, min(2.0, genome[j]))  # 限制在[-2, 2]
            next_population[i] = genome

        # 更新种群为下一代
        population = next_population

    # 遗传算法结束
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- 遗传算法结束 ---")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    print(f"找到的最佳适应度: {best_fitness_overall:.4f}")
    print(f"对应的最佳基因组: {best_genome_overall}")

    # 可以绘制适应度曲线 (需要 matplotlib)
    # import matplotlib.pyplot as plt
    # plt.plot(best_fitness_history, label="Best Fitness")
    # plt.plot(avg_fitness_history, label="Average Fitness")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness (Average Score)")
    # plt.legend()
    # plt.title("Fitness Evolution over Generations")
    # plt.savefig("fitness_evolution.png") # 保存图像
    # plt.show()

    return best_genome_overall


# ---------------------------
# 模型保存与应用
# ---------------------------

def save_best_genome(best_genome, filename="trained/optimized_genome_v2.pkl"):
    """将找到的最佳基因组保存到文件。"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(best_genome, file)
    print(f"最佳基因组已保存到: {filename}")


def load_best_genome2(filename="trained/optimized_genome_v2.pkl"):
    """从文件加载最佳基因组。"""
    if not os.path.exists(filename):
        print(f"错误: 找不到基因组文件 {filename}")
        return None
    try:
        with open(filename, 'rb') as file:
            best_genome = pickle.load(file)
        print(f"从 {filename} 加载基因组成功: {best_genome}")
        return best_genome
    except Exception as e:
        print(f"加载基因组时出错: {e}")
        return None


def Get_GA_Strategy2(best_genome, A_orig, B_orig):
    """
    使用最佳基因组，为给定的初始手牌A和B（按原始顺序处理B）
    演示决策过程并返回策略步骤。

    Args:
        best_genome (list): 训练好的最佳基因组。
        A_orig (list): 初始手牌A。
        B_orig (list): 初始手牌B (将按此列表顺序处理)。

    Returns:
        list: 包含策略步骤的列表，格式为 [(b_index, insert_pos, card_value), ...]
    """
    if best_genome is None:
        print("错误：基因组未提供或加载失败。")
        return []

    A_current = A_orig.copy()
    B_to_process = B_orig.copy()  # 使用传入的B顺序

    round_score = 0
    strategy_steps = []  # 存储策略 [(b_index, position, card_value)]



    # 按照 B_to_process 的顺序处理每一张牌
    for i, card_x in enumerate(B_to_process):


        # 使用基因组选择最佳插入位置
        pos, score, A_next = genome_choose_insertion(best_genome, A_current, card_x)


        # 累加得分并更新A的状态
        round_score += score
        A_current = A_next  # 更新A用于下一步决策

        # 记录这一步的策略
        strategy_steps.append((i, pos))



    return strategy_steps


# ---------------------------
# 主程序入口
# ---------------------------
if __name__ == "__main__":
    # # 设置遗传算法参数
    # POPULATION_SIZE = 500  # 种群大小 (可根据计算资源调整)
    # GENERATIONS = 40  # 迭代代数 (可根据需要调整)
    # NUM_ROUNDS_EVAL = 500  # 每代评估适应度的模拟轮数 (越高越准但越慢)
    # ELITISM_RATIO = 0.1  # 精英比例
    # TOURNAMENT_SIZE = 5  # 锦标赛选择大小
    # NUM_PROCESSES = 8  # 并行评估的进程数 (设为CPU核心数或稍小)
    #
    # # 运行遗传算法训练
    # best_genome_found = genetic_algorithm(
    #     pop_size=POPULATION_SIZE,
    #     generations=GENERATIONS,
    #     num_rounds=NUM_ROUNDS_EVAL,
    #     elitism_ratio=ELITISM_RATIO,
    #     tournament_size=TOURNAMENT_SIZE,
    #     num_processes=NUM_PROCESSES
    # )
    #
    #
    # save_best_genome(best_genome_found, filename="trained/optimized_genome_v2.pkl")

    # 加载并演示最佳基因组的效果 (可选)
    print("\n加载并测试最佳基因组...")
    loaded_genome = load_best_genome2(filename="trained/optimized_genome_v2.pkl")
    if loaded_genome:
        # 创建一副示例手牌用于演示
        test_A, test_B = deal_cards()
        # 获取策略演示
        Get_GA_Strategy2(loaded_genome, test_A, test_B)

