import random
import numpy as np
import time
import pickle
from multiprocessing import Pool



# ---------------------------
# 游戏规则函数
# ---------------------------

# 初始化一副牌（1到13，4种花色）
def init_deck():
    return [i for i in range(1, 14) for _ in range(4)]  # 创建一个包含四副牌的列表

# 洗牌并发牌，A玩家6张，B玩家3张
def deal_cards():
    deck = init_deck()  # 获取初始化的牌堆
    random.shuffle(deck)  # 洗牌
    a_unique = set()
   # A玩家抽取6张牌，这些牌不能有重复的 否则游戏规则混乱
    A = []
    while len(A) < 6:
        card = deck.pop()
        if card not in a_unique:
            a_unique.add(card)
            A.append(card)

    B = [deck.pop() for _ in range(3)]  # B玩家抽取3张牌
    return A, B  # 返回A和B玩家的牌
def get_best_insertion_score(A, card):
    max_score = -float('inf')
    best_pos = -1

    # 遍历所有可能的插入位置 (从位置1开始，因为位置0无效)
    for pos in range(1, len(A) + 1):
        score, _, _, _, _ = simulate_insertion(A, card, pos)
        
        # 找到最大的得分
        if score > max_score:
            max_score = score
            best_pos = pos

    return best_pos, max_score
def get_updated_A_after_insertion(A, card_x, best_pos):
    # 调用 simulate_insertion 来插入 x 到 A 的最佳位置并获取新的 A
    _, _, _, _, new_A = simulate_insertion(A, card_x, best_pos)
    return new_A

# def choose_insertion(genome, A, B, x, remaining_B):
# x是不能构成匹配的牌组成的数组
def calculate_future_score(A, remaining_B, genome, num_simulations=100):
    """
    使用蒙特卡罗方法估计未来得分
    Args:
        A: 当前A序列
        remaining_B: 剩余的B卡牌
        genome: 基因组权重
        num_simulations: 模拟次数
    Returns:
        float: 估计的未来期望得分
    """
    if len(remaining_B) == 0:
        return 0
    
    # 如果A和remaining_B没有重叠元素，直接返回0
    if not set(A) & set(remaining_B):
        return 0
        
    total_score = 0
    
    # 进行多次蒙特卡罗模拟
    for _ in range(num_simulations):
        # 复制当前状态以进行模拟
        current_A = A.copy()
        current_B = remaining_B.copy()
        simulation_score = 0
        
        # 随机打乱B卡牌的顺序来模拟不同的出牌顺序
        random.shuffle(current_B)
        
        # 模拟剩余的每一步
        for card in current_B:
            # 使用genome_choose_insertion来模拟决策
            remaining = [b for b in current_B if b != card]
            pos, score, new_A = genome_choose_insertion(genome, current_A, card, remaining)
            
            # 累加得分
            simulation_score += score
            current_A = new_A
            
        # 将本次模拟的得分加入总分
        total_score += simulation_score
    
    # 返回平均得分作为期望值
    expected_score = total_score / num_simulations
    
    return expected_score
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
def genome_choose_insertion(genome, A,  x, remaining_B):
    best_value, best_move = -float('inf'), None  # 初始化最佳评估值和最佳移动
    # pin=x
    # unmatched_cards = [card for card in remaining_B if card not in A]  # 找到B中在A中没有匹配的牌

    # if unmatched_cards:
    #     x = max(unmatched_cards)  # 如果存在未匹配的牌，优先选择其中值更大的牌

    possible_moves = []  # 存储所有候选插入位置的得分

    # 从位置0开始插入  现在可以从0开始了
    for pos in range(0, len(A) + 1):  # 尝试所有可能的插入位置
        score, removal_length, new_length, match_found, new_A = simulate_insertion(A, x, pos)  # 模拟插入并获取结果
        # print(f"x is {x}")
        current_score = score  # 当前得分

        future_score = calculate_future_score(new_A, remaining_B, genome)  # 计算未来的得分

        features = np.array([current_score, removal_length, new_length, match_found, future_score], dtype=float)  # 特征向量
        value = np.dot(genome, features)  # 基因组的评估值

        possible_moves.append((value, pos, current_score, new_A))  # 将当前插入位置及其得分保存到列表中

    # 如果对于特定的B元素 有多个可能的插入位置，选择得分最高的那个插入位置
    if possible_moves:
        best_move = max(possible_moves, key=lambda move: move[0])

    # 当 A 为空时 ，此时无法进行任何插入操作，导致 possible_moves 为空，进而 best_move 未被更新，保持初始值 None
    if best_move is None:
        # possible_moves = []
        # for pos in range(1, len(A) + 1):  # 尝试所有可能的插入位置
        #     score, removal_length, new_length, match_found, new_A = simulate_insertion(A, x, pos)  # 模拟插入并获取结果
        #     possible_moves.append((score, pos, new_A))  # 将当前插入位置及其得分保存到列表中

        # 默认插入到末尾，否则没有返回值

        return len(A), 0, A

        # best_move = max(possible_moves, key=lambda move: move[0])

    pos, score, new_A = best_move[1], best_move[2], best_move[3]  # 提取最佳插入位置、得分和新牌堆
    return pos, score, new_A  # 返回最佳插入位置、得分和新牌堆



# 模拟一轮游戏
def simulate_round(genome):

    A, B = deal_cards()  # 发牌
    round_score = 0  # 初始化本轮得分
    for i, x in enumerate(B):  # 遍历B玩家的每一张牌
        remaining_B = B[i+1:]  # 获取后续未处理的B牌
        pos, score, A = genome_choose_insertion(genome, A, x, remaining_B)  # 选择最优插入位置并更新牌堆
        round_score += score  # 累加得分
    return round_score  # 返回本轮总得分

# 评估基因组的适应度
def evaluate_genome(genome, num_rounds=1000):
    scores = [simulate_round(genome) for _ in range(num_rounds)]  # 运行多次模拟并记录得分
    return np.median(scores) * 0.7 + np.mean(scores) * 0.3  # 计算加权平均得分

# 使用多进程评估基因组适应度
def evaluate_genomes_with_processes(population, num_rounds=1000, num_processes=8):
    with Pool(processes=num_processes) as pool:  # 创建进程池
        fitnesses = pool.starmap(evaluate_genome, [(genome, num_rounds) for genome in population])  # 并行评估每个基因组
    return fitnesses  # 返回评估结果

# 遗传算法过程
def genetic_algorithm(pop_size=1000, generations=60, num_rounds=1000, elitism_ratio=0.1, tournament_size=3, num_processes=8):

    start_time = time.time()  # 记录开始时间
    population = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(pop_size)]  # 初始化种群
    best_fitness_history, avg_fitness_history = [], []  # 初始化历史最佳适应度和平均适应度列表
    best_genome, best_fitness = None, -float('inf')  # 初始化最佳基因组和最佳适应度
    elitism_count = int(elitism_ratio * pop_size)  # 计算精英数量
    
    for gen in range(generations):  # 迭代每个世代
        fitnesses = evaluate_genomes_with_processes(population, num_rounds, num_processes)  # 评估种群适应度
        gen_best, gen_avg = max(fitnesses), np.mean(fitnesses)  # 获取当前世代的最佳适应度和平均适应度
        best_fitness_history.append(gen_best)  # 记录最佳适应度
        avg_fitness_history.append(gen_avg)  # 记录平均适应度
        
        if gen_best > best_fitness:  # 更新最佳基因组和适应度
            best_fitness, best_genome = gen_best, population[fitnesses.index(gen_best)]
        
        print(f"Generation {gen+1}: Best Fitness = {gen_best:.2f}, Average Fitness = {gen_avg:.2f}")  # 打印当前世代信息
        
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]  # 按适应度排序种群
        elites = sorted_population[:elitism_count]  # 选择精英
        
        selected = [population[max(random.sample(range(pop_size), tournament_size), key=lambda i: fitnesses[i])] for _ in range(pop_size - elitism_count)]  # 通过锦标赛选择剩余的基因组
        
        next_population = elites.copy()  # 初始化下一代种群
        while len(next_population) < pop_size:  # 生成下一代
            parent1, parent2 = random.sample(selected, 2)  # 选择两个父本
            child = [(p1 + p2) / 2 if random.random() < 0.7 else p1 for p1, p2 in zip(parent1, parent2)]  # 交叉产生子代
            mutation_rate = max(0.1, 0.3 - (gen / generations) * 0.2)  # 计算变异率
            child = [gene + random.gauss(0, 0.5) if random.random() < mutation_rate else gene for gene in child]  # 变异
            next_population.append(child)  # 添加子代到下一代种群
        
        population = next_population  # 更新种群
    
    end_time = time.time()  # 记录结束时间
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")  # 打印总执行时间
    
    return best_genome  # 返回最佳基因组


def save_best_genome(best_genome, filename="trained/best_genome.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(best_genome, file)  # 保存最佳基因组到文件
    print(f"Best genome saved to {filename}")  # 打印保存信息
def GA(genome,A,B):
    round_score = 0  # 初始化本轮得分
    for i, x in enumerate(B):  # 遍历B玩家的每一张牌
        remaining_B = B[i + 1:]  # 获取后续未处理的B牌
        pos, score, A = genome_choose_insertion(genome, A, x, remaining_B)  # 选择最优插入位置并更新牌堆
        round_score += score  # 累加得分
    return round_score  # 返回本轮总得分
def GA_Strategy(best_genome, A, B):

    round_score = 0
    num_moves = 0
    strategy = []  # 用于存储i和pos的值
    print("-" * 20)
    for i, x in enumerate(B):
        remaining_B = B[i+1:]
        pos, score, A = genome_choose_insertion(best_genome, A, x, remaining_B)

        # # 打印移动信息
        # print(f"Move {num_moves + 1}: Insert card {x} at position {pos}")

        # 更新得分和移动次数
        round_score += score
        num_moves += 1

        # 保存i和pos的值
        strategy.append((i, pos))

    # print(f"Turn Score: {round_score}, Total Moves: {num_moves}")

    return strategy  # 将原始的A，B和最优插牌得分策略返回

if __name__ == "__main__":

    best_genome = genetic_algorithm()  # 运行遗传算法获取最佳基因组
    print("\nGenome model : ", best_genome)  # 打印最佳基因组
    # evaluate_final_model(best_genome)  # 评估最终模型性能
    save_best_genome(best_genome)  # 保存最佳基因组
    #
    # genome=load_best_genome()
    # GA_partial = partial(GA, genome)
    # modsummery(GA_partial,2000)



