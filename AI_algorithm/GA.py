import multiprocessing
import random
import numpy as np
import time
import pickle
import scipy.linalg
import json
import os
import torch

from AI_algorithm.tool.tool import calculate_future_score, simulate_insertion_tool


# ---------------------------
# 游戏规则函数
# ---------------------------

# 初始化一副牌（1到13，4种花色）
def init_deck():
    return [i for i in range(1, 14) for _ in range(4)]  # 创建一个包含四副牌的列表


# 在文件顶部添加全局缓存
_json_cache = {}

def deal_cards(json_file="AI_algorithm/json/data_raw_2.json", seed=None):
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        random.seed(seed)

    # 检查缓存中是否已有数据
    if json_file in _json_cache:
        cases = _json_cache[json_file]
    else:
        # 检查文件是否存在
        full_path = os.path.join(os.path.dirname(__file__), "json", os.path.basename(json_file))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"文件未找到: {full_path}")

        try:
            # 从JSON文件读取数据
            with open(full_path, 'r') as f:
                cases = json.load(f)
            # 存入缓存
            _json_cache[json_file] = cases
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"读取JSON文件时出错: {e}")
            raise

    # 随机选择一个案例
    case = random.choice(cases)

    # 从案例中提取A和B的牌
    A = case.get('A', [])
    B = case.get('B', [])

    # 如果JSON中没有提供牌，则使用默认的随机发牌逻辑
    if not A or not B:
        raise ValueError("A或B为空")

    return A, B




# 评估基因组的适应度  还是贪心算法  但是至少不是穷举
def GA_Strategy(genome, A, B):
    """
    使用基因组评估所有可能的B牌处理顺序，返回最高得分和相应策略

    参数：
    - genome: 基因组权重 (NumPy数组或PyTorch张量)
    - A: A玩家的牌
    - B: B玩家的牌

    返回：
    - strategy: 最佳策略
    """
    import torch
    import numpy as np

    # 检查CUDA可用性并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查genome类型并转换为torch张量
    if isinstance(genome, torch.Tensor):
        genome_tensor = genome.to(device)
    else:
        genome_tensor = torch.tensor(genome, dtype=torch.float32, device=device)



    # 批量计算所有B牌的所有可能插入位置的特征
    def compute_card_values_batch(A_copy, B, genome_tensor):
        card_values = []

        # 预先计算一些共用特征
        sum_A = sum(A_copy)
        A_set = set(A_copy)

        for i, card in enumerate(B):
            remaining_B = [B[j] for j in range(len(B)) if j != i]
            remaining_B_set = set([card] + remaining_B)
            intersection_count = len(remaining_B_set & A_set)

            # 为所有可能的插入位置创建特征批次
            all_features = []
            all_positions = []


            # 收集所有位置的特征
            for pos in range(len(A_copy) + 1):
                score, removal_length, new_length, match_found, new_A = simulate_insertion_tool(A_copy, card, pos)
                future_score = calculate_future_score(new_A, remaining_B)

                features = [
                    score,  # 当前得分
                    removal_length,  # 移除长度
                    new_length,  # 新长度
                    future_score,  # 未来得分
                    sum_A,  # A的元素总和
                    intersection_count,  # B与A的交集数量
                ]

                all_features.append(features)
                # 最佳插入位置
                all_positions.append(pos)
                # all_scores.append(score)
                # all_new_As.append(new_A)

            # 将所有特征转换为GPU张量并一次性计算
            if all_features:
                features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
                # 批量计算点积 (每个位置的评估值)
                values = torch.matmul(features_tensor, genome_tensor)

                # 找到最佳位置
                best_idx = torch.argmax(values).item()
                best_value = values[best_idx].item()
                best_pos = all_positions[best_idx]
                # best_score = all_scores[best_idx]
                # best_new_A = all_new_As[best_idx]

                card_values.append((i, best_value, best_pos))

        return card_values

    # 计算卡值
    card_values = compute_card_values_batch(A.copy(), B, genome_tensor)

    # 根据评估值对卡进行排序 x[1] 就是 best_value ，即每张牌的最佳评估值
    card_values.sort(key=lambda x: x[1], reverse=True)

    # 根据排序后的顺序生成策略
    strategy = [(card_idx, pos) for card_idx, _, pos in card_values]

    return strategy

# 每一个generation执行pop_size次
def evaluate_genome(genome, num_rounds=1000, seed_base=111):
    import torch
    import numpy as np
    from AI_algorithm.tool.tool import calculate_score_by_strategy

    # 检查并设置 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 将基因组转换为 GPU 张量
    # genome_tensor = torch.tensor(genome, dtype=torch.float32, device=device)

    # 批处理大小
    batch_size = 32

    # 预分配结果数组
    total_scores = torch.zeros(num_rounds, dtype=torch.float32, device=device)

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

            strategy = GA_Strategy(genome, batch_A[j], batch_B[j])
            batch_scores[j] = calculate_score_by_strategy(batch_A[j], batch_B[j], strategy)

        # 复制批次得分到总得分
        total_scores[batch_start:batch_end] = batch_scores

    # 计算最终平均分
    mean_score = torch.mean(total_scores).item()

    print(f"基因组平均得分: {mean_score}")

    return mean_score  # 返回平均得分
# def evaluate_genomes_return_fitness(population, num_rounds=1000):
#
#
#
#     fitnesses = []
#
#
#     for genome in population:
#         try:
#             # print(f"正在评估第 {population.index(genome) + 1} /{len(population)}个基因组")
#             fitness = evaluate_genome(genome, num_rounds)
#
#             fitnesses.append(fitness)
#         except Exception as e:
#             print(f"基因组 {genome} 评估发生异常: {e}")
#             fitnesses.append(-float('inf'))
#
#     return fitnesses


def evaluate_genomes_return_fitness(population, num_rounds):
    """并行评估多个基因组"""

    # 创建进程池
    with multiprocessing.Pool(8) as pool:
        # 准备评估参数
        eval_args = [(genome, num_rounds) for genome in population]
        # 并行计算
        fitnesses = pool.starmap(evaluate_genome, eval_args)

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




# 遗传算法过程
def genetic_algorithm(pop_size, generations, num_rounds, elitism_ratio, tournament_size,
                      evolution_methods,
                      early_stop_generations, early_stop_threshold):
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
        genome = []
        # 引入更多随机性
        if random.random() < 0.7:  # 70%的个体按原方式初始化
            for i in range(6):
                if i == 0 or i == 3:
                    genome.append(random.uniform(0, 2))
                else:
                    genome.append(random.uniform(-1, 1))
        else:  # 30%的个体完全随机初始化
            for i in range(6):
                genome.append(random.uniform(-2, 2))
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
            if early_stop_counter > 0:
                mutation_rate += 0.1 * early_stop_counter  # 每停滞一代增加10%的变异率
            mutation_rate = min(mutation_rate, 0.8)  # 但不超过80%
            child = [gene + random.gauss(0, 0.5) if random.random() < mutation_rate else gene for gene in child]  # 变异
            next_population.append(child)  # 添加子代到下一代种群

        # # 根据概率选择进化方法
        # method_idx = np.random.choice(len(evolution_methods), p=method_probs)
        # method = evolution_methods[method_idx]
        # method_history.append(method)

        if len(best_fitness_history) > 1:
            improvement = best_fitness_history[-1] - best_fitness_history[-2]
            if improvement > 0:
                method = method_history[-1]
                print(f"改进率为 {improvement:.4f}，沿用方法：{method}")
            else:
                method = np.random.choice(evolution_methods)
                print(f"改进率为 {improvement:.4f}，切换方法为：{method}")
        else:
            method = np.random.choice(evolution_methods)
            print(f"无历史改进数据，随机选择方法：{method}")

        method_history.append(method)

        if method == 'standard':
            pass  # 使用标准遗传算法
        elif method == 'island':
            next_population, _ = island_model_evolution(next_population, fitnesses, pop_size, tournament_size, 0.5, num_rounds, generation=gen, max_generations=generations)
        elif method == 'de':
            next_population, _ = differential_evolution(next_population, fitnesses, pop_size, F=0.8, CR=0.5, num_rounds=num_rounds, generation=gen, max_generations=generations)

        population = next_population  # 更新种群

    end_time = time.time()  # 记录结束时间
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")  # 打印总执行时间
    print(f"Completed Generations: {gen + 1} of {generations}")  # 打印实际完成的代数

    # 分析不同进化方法的性能
    analyze_evolution_methods(best_fitness_history, method_history,evolution_methods)

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
def analyze_evolution_methods(best_fitness_history, method_history, all_methods):
    """
    分析不同进化方法的性能，确保所有方法都会被分析
    """
    # 初始化所有方法的性能列表
    method_performance = {method: [] for method in all_methods}

    # 按照进化方法分组
    for method, fitness in zip(method_history, best_fitness_history):
        method_performance[method].append(fitness)

    # 计算每种方法的平均性能和最大性能
    print("\n各进化方法性能分析:")
    for method, fitnesses in method_performance.items():
        if fitnesses:  # 如果该方法被使用过
            avg_fitness = sum(fitnesses) / len(fitnesses)
            max_fitness = max(fitnesses)
            improve_rate = (fitnesses[-1] - fitnesses[0]) / fitnesses[0] if len(fitnesses) > 1 else 0
            print(
                f"{method}方法: 平均适应度={avg_fitness:.2f}, 最大适应度={max_fitness:.2f}, 改进率={improve_rate:.2%}, 使用次数={len(fitnesses)}")
        else:  # 如果该方法未被使用
            print(f"{method}方法: 未被使用")
#


if __name__ == "__main__":

    pop_size=300
    generations=30
    num_rounds=80
    elitism_ratio=0.1
    tournament_size=3
    evolution_methods=['standard', 'island', 'de']
    # method_probs=[0.4, 0.3, 0.3]
    early_stop_generations=10
    early_stop_threshold=0.01

    genome = genetic_algorithm(pop_size,generations, num_rounds, elitism_ratio, tournament_size, evolution_methods, early_stop_generations, early_stop_threshold)  # 运行遗传算法获取最佳基因组
    print("\nGenome model : ", genome)  # 打印最佳基因组
    # evaluate_final_model(genome)  # 评估最终模型性能
    save_best_genome(genome)  # 保存最佳基因组