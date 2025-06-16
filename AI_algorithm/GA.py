import multiprocessing

import numpy as np
import time
import pickle
import json
import os


from AI_algorithm.tool.tool import calculate_future_score, simulate_insertion_tool

# Todo 训练前注意路径!!!
jsonfile_path = "AI_algorithm/json/data_GA_fill.json"
genome_path = "trained/best_genome.pkl"

#
# jsonfile_path = "AI_algorithm/json/data_GA_skip.json"
# genome_path="trained/best_genome_skip.pkl"



# 初始化一副牌（1到13，4种花色）
def init_deck():
    return [i for i in range(1, 14) for _ in range(4)]  # 创建一个包含四副牌的列表


# 在文件顶部添加全局缓存
_json_cache = {}

def deal_cards(json_file=jsonfile_path, seed=None):
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

        #特征计算


        #使用嵌套循环进行插入模拟  每一个插入位置都会计算6个特征
        for i, card in enumerate(B):
            remaining_B = [B[j] for j in range(len(B)) if j != i]
            #remaining_B_set = set([card] + remaining_B)
            #特征计算

            B_card_proportion  = card/sum(remaining_B)

            # 插入位置以及特征集合
            all_features = []
            all_positions = []

            for pos in range(len(A_copy) + 1):
                #特征计算
                score, removal_length, new_length, _, new_A = simulate_insertion_tool(A_copy, card, pos)
                sum_new_A = sum(new_A)
                future_score = calculate_future_score(new_A, remaining_B)

                # 需要标准化
                features = [
                    score,  # 当前得分
                    removal_length,  # 移除长度
                    new_length,  # 新长度
                    future_score,  # 未来得分
                    sum_new_A,  # 匹配后新A的元素总和
                    B_card_proportion,  # B的出牌占B剩余牌的价值比例
                ]

                all_features.append(features)

                all_positions.append(pos)
                # all_scores.append(score)
                # all_new_As.append(new_A)

            # 将所有特征转换为张量并一次性在GPU上计算
            if all_features:
                features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

                # --- 關鍵步驟：在此處添加批次標準化 ---
                # 1. 計算當前批次特徵的均值和標準差
                #    keepdim=True 保持維度以便廣播
                mean = torch.mean(features_tensor, dim=0, keepdim=True)
                std = torch.std(features_tensor, dim=0, keepdim=True)

                # 2. 為防止除以零，給標準差加上一個極小值 (epsilon)
                epsilon = 1e-8

                # 3. 進行標準化 (Z-score)
                standardized_features = (features_tensor - mean) / (std + epsilon)
                # ----------------------------------------

                # 使用標準化後的特徵進行加權求值
                values = torch.matmul(standardized_features, genome_tensor)

                best_idx = torch.argmax(values).item()
                best_value = values[best_idx].item()
                best_pos = all_positions[best_idx]

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
def evaluate_genome(genome, num_rounds, seed_base):
    import torch
    from AI_algorithm.tool.tool import calculate_score_by_strategy

    # 检查并设置 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 将基因组转换为 GPU 张量
    # genome_tensor = torch.tensor(genome, dtype=torch.float32, device=device)

    # 批处理大小
    batch_size = 1024

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



import functools  # <== 在文件顶部导入


# 函数签名完全不变！
def evaluate_genomes_return_fitness(population, num_rounds):
    """并行评估多个基因组"""

    # ====================【核心修改】====================
    # 1. 准备一个“空白”的评估函数
    #    这个函数缺少 seed_base 参数，我们不能直接调用它
    base_eval_func = evaluate_genome

    # 2. 我们需要一个每一代都不同的种子
    #    这里我们用一个简单的方法获取一个相对随机但可控的种子
    #    用当前时间和一个大的随机数结合，确保每次运行的种子序列都不同
    #    这可以防止整个GA对某一个固定的种子序列过拟合
    import time
    generation_seed_base = int(time.time() * 1000) + random.randint(0, 10000)

    # 3. 使用 partial 创建一个新函数
    #    这个新函数 `eval_func_with_seed` 就像是 evaluate_genome，
    #    但是它的 `seed_base` 参数已经被我们用 `generation_seed_base` 永久地固定住了。
    eval_func_with_seed = functools.partial(base_eval_func, seed_base=generation_seed_base)

    # 4. 在并行池中使用这个“预先绑定好种子”的新函数
    #    我们只需要传递剩下的参数 (genome, num_rounds)
    with multiprocessing.Pool(8) as pool:
        eval_args = [(genome, num_rounds) for genome in population]
        # 注意：这里调用的是我们刚创建的 eval_func_with_seed
        fitnesses = pool.starmap(eval_func_with_seed, eval_args)
    # =====================================================

    return fitnesses

# 岛屿模型实现，用于增加种群多样性
import random


def island_model_evolution(population, fitnesses, pop_size, tournament_size, mutation_strength, islands=3,
                           migration_rate=0.1):
    """
    【高效版】实现岛屿模型进化。
    该函数只负责生成下一代种群，不进行任何适应度评估。
    迁移策略：每个岛屿随机选择个体进行迁移，替换目标岛屿中适应度最差的个体。

    参数：
    - population: 当前种群 (list)
    - fitnesses: 当前种群的适应度值 (list)
    - pop_size: 总人口规模 (int)
    - tournament_size: 锦标赛选择规模 (int)
    - mutation_strength: 变异强度 (float)
    - islands: 岛屿数量 (int)
    - migration_rate: 迁移率 (float)

    返回：
    - new_population: 进化后的新种群 (list)
    """

    # 1. 分割岛屿
    # =================================================================
    island_size = pop_size // islands
    island_populations = []
    island_fitnesses = []
    for i in range(islands):
        start_idx = i * island_size
        end_idx = start_idx + island_size if i < islands - 1 else pop_size
        island_populations.append(population[start_idx:end_idx])
        island_fitnesses.append(fitnesses[start_idx:end_idx])

    # 2. 每个岛屿独立进化，产生下一代候选种群
    # =================================================================
    new_island_populations = []
    for i in range(islands):
        # 使用传入的 fitnesses[i] 进行精英选择和锦标赛选择
        sorted_indices = sorted(range(len(island_fitnesses[i])),
                                key=lambda k: island_fitnesses[i][k], reverse=True)

        elitism_count = int(0.1 * len(island_populations[i]))
        elites = [island_populations[i][idx] for idx in sorted_indices[:elitism_count]]

        selected = []
        # 修正：锦标赛选择的候选者应来自整个岛屿，而不是非精英
        for _ in range(len(island_populations[i]) - elitism_count):
            candidates_indices = random.sample(range(len(island_populations[i])), tournament_size)
            winner_idx = max(candidates_indices, key=lambda idx: island_fitnesses[i][idx])
            selected.append(island_populations[i][winner_idx])

        # 交叉和变异
        next_population = elites.copy()
        while len(next_population) < len(island_populations[i]):
            parent1, parent2 = random.sample(selected, 2)
            child = [(p1 + p2) / 2 if random.random() < 0.7 else p1 for p1, p2 in zip(parent1, parent2)]
            child = [gene + random.gauss(0, mutation_strength) for gene in child]
            next_population.append(child)

        new_island_populations.append(next_population)

    # 🛑 移除了第一次评估调用

    # 3. 迁移过程
    # =================================================================
    migration_size = int(island_size * migration_rate)
    if migration_size == 0:  # 确保至少有1个迁移者，如果种群很小
        migration_size = 1

    # 使用一个副本进行迁移操作，避免在迭代时修改列表
    migrated_populations = [pop[:] for pop in new_island_populations]

    for i in range(islands):
        # 从当前新生成的岛屿中，随机选择迁移者
        if len(migrated_populations[i]) > migration_size:
            migrant_indices = random.sample(range(len(migrated_populations[i])), migration_size)
            migrants = [migrated_populations[i][idx] for idx in migrant_indices]
        else:  # 如果岛屿太小，则全部迁移
            migrants = migrated_populations[i][:]

        # 确定目标岛屿
        target_island_idx = (i + 1) % islands

        # 在目标岛屿中，找到要被替换的个体
        # 关键：我们使用进化前的适应度 `island_fitnesses` 来决定谁最差
        target_fitnesses = island_fitnesses[target_island_idx]
        sorted_target_indices = sorted(range(len(target_fitnesses)), key=lambda k: target_fitnesses[k])

        # 替换最差的个体
        for j in range(min(migration_size, len(sorted_target_indices))):
            replace_idx = sorted_target_indices[j]
            if j < len(migrants):
                migrated_populations[target_island_idx][replace_idx] = migrants[j]

    new_island_populations = migrated_populations

    # 🛑 移除了第二次评估调用

    # 4. 合并所有岛屿种群
    # =================================================================
    new_population = []
    for pop in new_island_populations:
        new_population.extend(pop)

    # 🛑 移除了基于适应度的排序和裁剪，因为我们没有新适应度。
    #    裁剪可以在主循环评估后进行，或者直接返回合并后的种群。
    #    这里我们直接返回，让主循环决定如何处理超额部分。
    if len(new_population) > pop_size:
        new_population = new_population[:pop_size]

    # 只返回新种群
    return new_population


# 差分进化算法实现(不使用)
def differential_evolution(population, fitnesses, pop_size, F=0.8, CR=0.5, num_rounds=1000,
                           generation=0, max_generations=60):
    """
    实现差分进化算法 (Differential Evolution Algorithm)

    差分进化是一种基于种群的全局优化算法，通过变异、交叉、选择三个步骤
    不断改进种群质量，寻找问题的最优解。

    算法特点：
    - 利用种群个体间的差异信息指导搜索方向
    - 自适应调整参数以平衡全局探索和局部开发
    - 采用贪心选择策略确保种群质量单调提升

    参数说明：
    - population: 当前种群，二维列表，每个子列表代表一个个体的基因序列
    - fitnesses: 当前种群的适应度值列表，数值越小表示个体越优秀（最小化问题）
    - pop_size: 种群规模，即个体数量
    - F: 缩放因子/变异因子 (典型值:0.5-1.0)
         控制变异向量的步长大小，值越大变异幅度越大
    - CR: 交叉概率 (典型值:0.1-0.9)
          控制试验向量从变异向量继承基因的概率
    - num_rounds: 适应度评估轮数，用于fitness函数的参数
    - generation: 当前进化代数，用于参数自适应调整
    - max_generations: 最大进化代数，用于计算进化进度

    返回值：
    - new_population: 进化后的新种群
    - new_fitnesses: 新种群对应的适应度值
    """
    # 动态调整F和CR参数
    # progress_ratio需要＞0
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
        """
                        基于先验知识的种群初始化策略


                        这里使用了一个优化手段足够多的测试后 ，经过了足够多的测试后,可以得出以下结论：
                        基因特征中,score和future_score对于策略的正面影响较大 在1.5以上 
        """
        if random.random() < 0.7:  # 70%的个体按原方式初始化
            for i in range(6):
                if i == 0 or i == 3 :
                    genome.append(random.uniform(1.5, 2.5))
                else:
                    genome.append(random.uniform(-0.5, 0.5))
        else:  # 30%的个体完全随机初始化
            for i in range(6):
                # 用[-2,2]随机浮点数
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
        print(f"\n--- [Generation {gen + 1}/{generations}] Evaluating population... ---")
        fitnesses = evaluate_genomes_return_fitness(population, num_rounds)  # 评估种群适应度
        gen_best, gen_avg = max(fitnesses), np.mean(fitnesses)  # 获取当前世代的最佳适应度和平均适应度
        best_fitness_history.append(gen_best)  # 记录最佳适应度
        avg_fitness_history.append(gen_avg)  # 记录平均适应度

        # 早停检查
        improvement = gen_best - last_best_fitness

        # 修复除法错误 确保 last_best_fitness 不是无效值
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



        if len(best_fitness_history) > 1:
            improvement = best_fitness_history[-1] - best_fitness_history[-2]
            if improvement > 0:
                method = method_history[-1]
                print(f"改进率为 {improvement:.4f}，沿用方法：{method}")
            else:
                method = method_history[-1]
                # 找到 现在方法 的索引
                index_of_method_now = evolution_methods.index(method)

                # 获取另一个方法
                method = evolution_methods[1 - index_of_method_now]
                #method = np.random.choice(evolution_methods)
                print(f"改进率为 {improvement:.4f}，切换方法为：{method}")
        else:
            method = evolution_methods[0]
            print(f"无历史改进数据，使用第一个方法：{method}")

        method_history.append(method)

        if method == 'standard':
            pass  # 使用标准遗传算法
        elif method == 'island':
            next_population= island_model_evolution(next_population, fitnesses, pop_size, tournament_size, 0.7)
        # elif method == 'de':
        #     next_population, _ = differential_evolution(next_population, fitnesses, pop_size, F=0.8, CR=0.5, num_rounds=num_rounds, generation=gen, max_generations=generations)

        population = next_population  # 更新种群

    end_time = time.time()  # 记录结束时间
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")  # 打印总执行时间
    print(f"Completed Generations: {gen + 1} of {generations}")  # 打印实际完成的代数

    # 分析不同进化方法的性能
    analyze_evolution_methods(best_fitness_history, method_history,evolution_methods)

    return best_genome  # 返回最佳基因组


def save_best_genome(genome, filename=genome_path):
    with open(filename, 'wb') as file:
        pickle.dump(genome, file)  # 保存最佳基因组到文件
    print(f"Best genome saved to {filename}")  # 打印保存信息

    # 打印基因组各特征的权重
    feature_names = [
        "当前得分", "移除长度", "新长度",  "未来得分",
        "新A元素总和",  "出牌占B剩余牌的价值比例"
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
    # Todo 训练前注意路径!!!

    pop_size=500
    generations=25
    num_rounds=500
    elitism_ratio=0.1
    tournament_size=3
    evolution_methods=['island','standard']

    early_stop_generations=3
    early_stop_threshold=0.01

    genome = genetic_algorithm(pop_size,generations, num_rounds, elitism_ratio, tournament_size, evolution_methods, early_stop_generations, early_stop_threshold)  # 运行遗传算法获取最佳基因组
    print("\nGenome model : ", genome)  # 打印最佳基因组
    # evaluate_final_model(genome)  # 评估最终模型性能
    save_best_genome(genome)  # 保存最佳基因组