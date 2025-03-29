import pickle
import time
import numpy as np
import torch
from scipy import stats
from functools import partial

from sympy import false

from AI_algorithm.Deep_Neural_Network import  DNNpredict
from AI_algorithm.GA import genome_choose_insertion



import matplotlib.pyplot as plt

from AI_algorithm.brute_force import recursive_strategy
from AI_algorithm.server import load_model_from_memory, MODEL_PATH
from AI_algorithm.tool.tool import load_best_genome, deal_cards_tool
from AI_algorithm.transformer_score import transformer_scoreonly

genome_loaded = load_best_genome()
def  testGeonme_with_given_A_B(A, B, best_genome=genome_loaded):

        
    # 复制 A 避免修改原始数据
    genetic_A = A.copy()
    genetic_score = 0

    # 计算遗传算法的运行时间
    start_time_genetic = time.time()
    for i, card in enumerate(B):
        remaining_B = B[i + 1 :]
        _, score, genetic_A = genome_choose_insertion(best_genome, genetic_A, card, remaining_B)
        genetic_score += score
    end_time_genetic = time.time()
    time_genetic = end_time_genetic - start_time_genetic  # 计算遗传算法时间

    # # 计算 other_model 算法的运行时间
    # start_time_other = time.time()
    # other_score = other_model(A.copy(), B.copy())
    # end_time_other = time.time()
    # time_other = end_time_other - start_time_other  # 计算其他算法时间

    return genetic_score, time_genetic
def TestOther_with_given_A_B(other_model,A,B):
    # 计算 other_model 算法的运行时间
    start_time_other = time.time()
    other_score = other_model(A.copy(), B.copy())
    end_time_other = time.time()
    time_other = end_time_other - start_time_other  # 计算其他算法时间
    return other_score, time_other
def compute_statistics(data):
    """计算均值、中位数、标准差、最小值和最大值"""
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data)

    }

def print_statistics(name, stats_dict):
    """打印统计信息"""
    print(f"【{name}】")
    print(f"  平均得分      : {stats_dict['mean']:.2f}")
    print(f"  中位数得分    : {stats_dict['median']:.2f}")
    print(f"  得分标准差    : {stats_dict['std_dev']:.2f}")
    print(f"  最低得分      : {stats_dict['min']}")
    print(f"  最高得分      : {stats_dict['max']}\n")

def compute_statistical_tests(data_of_algorithm_1, data_recursive):
    """执行配对 t 检验和 Cohen's d"""
    score_diff = data_of_algorithm_1 - data_recursive
    diff_stats = compute_statistics(score_diff)

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(data_of_algorithm_1, data_recursive)
    significance = "显著 (p < 0.05)" if p_value < 0.05 else "不显著 (p ≥ 0.05)"

    # 计算 Cohen's d
    cohen_d = diff_stats['mean'] / diff_stats['std_dev'] if diff_stats['std_dev'] != 0 else 0.0
    effect_size_label = (
        "极小" if abs(cohen_d) < 0.2 else
        "较小" if abs(cohen_d) < 0.5 else
        "中等" if abs(cohen_d) < 0.8 else
        "较大"
    )

    # print_statistics("以下是各项数据得分差异", diff_stats)
    print(f"  均值偏差      : {diff_stats['mean']:.2f}")
    print(f"  中位数偏差     : {diff_stats['median']:.2f}")
    print(f"  标准差偏差    : {diff_stats['std_dev']:.2f}")
    print(f"  最低得分偏差      : {diff_stats['min']}")
    print(f"  最高得分偏差      : {diff_stats['max']}\n")


    print("【配对 t 检验】")
    print(f"  t 统计量      : {t_stat:.3f}")
    print(f"  p 值          : {p_value:.3e}")
    print(f"  统计显著性    : {significance}\n")

    print("【效应量分析 (Cohen's d)】")
    print(f"  Cohen's d      : {cohen_d:.3f}")
    print(f"  影响程度      : {effect_size_label}\n")




def Compare_TwoModel(model, other_model, rounds=1000,plot=false):
    print("开始Compare_TwoModel测试...")  # 确保函数被调用

    test_scores_1 = []
    test_scores_2 = []
    true_score_list = []
    times_list1 = []
    times_list2 = []

    # 保存每一轮的得分
    all_scores_1 = []
    all_scores_2 = []
    all_true_scores = []

    for i in range(rounds):
        print(f"第 {i + 1} 轮测试")  # 观察是否卡在某一轮
        # 不应该使用无法得分的A,B训练
        while True:
            A, B = deal_cards_tool()  # 初始A, B   A, B 都是 list<int>

            # 检查 A 和 B 是否有任何重复的元素
            if  set(A) & set(B):  # 如果 A 和 B 有重复元素
                break  # 退出循环，继续处理这对 A, B
        print("  生成 A, B 成功")
        print(A)
        print(B)

        score_1, time_1 = TestOther_with_given_A_B(model, A, B)


        times_list1.append(time_1)


        score_2, time_2 = TestOther_with_given_A_B(other_model, A, B)

        times_list2.append(time_2)

        true_score, _ = TestOther_with_given_A_B(recursive, A, B)
        print('-'*20)
        print(f" 真实值 ：{true_score}")
        print(f" 算法1得分 ：{score_1}")
        print(f" 算法2得分 ：{score_2}")
        print('-' * 20)
        test_scores_1.append(score_1)
        test_scores_2.append(score_2)
        true_score_list.append(true_score)

        # 保存本轮的得分数据
        all_scores_1.append(score_1)
        all_scores_2.append(score_2)
        all_true_scores.append(true_score)

    print("所有轮次计算完成，开始统计数据...")

    data_1 = np.array(test_scores_1)
    data_2 = np.array(test_scores_2)
    data_true = np.array(true_score_list)

    times_1 = np.array(times_list1)
    times_2 = np.array(times_list2)

    stats_1 = compute_statistics(data_1)
    stats_2 = compute_statistics(data_2)
    stats_true = compute_statistics(data_true)

    stats_times_1 = np.sum(times_1)
    stats_times_2 = np.sum(times_2)

    print("统计信息计算完成，准备输出...")

    print_statistics("算法1", stats_1)
    print_statistics("算法2", stats_2)
    print_statistics("真实值", stats_true)

    print(f"""【运行时间】
    算法1运行时间: {stats_times_1}
    算法2运行时间: {stats_times_2}
    """)

    # 计算均方误差（MSE）
    mse_1 = np.mean((data_1 - data_true) ** 2)
    print(f"【均方误差（MSE）】")
    print(f"算法1与真实值的均方误差: {mse_1}")

    mse_2 = np.mean((data_2 - data_true) ** 2)
    print(f"【均方误差（MSE）】")
    print(f"算法2与真实值的均方误差: {mse_2}")

    # 统计检验分析
    print("-" * 20)
    print("\n【算法1与真实值的偏差分析】\n")
    print("*所有偏差均按照“真实值相应数据减去预测值相应数据”的方式计算")
    compute_statistical_tests(data_1, data_true)
    print("-" * 20)
    print("\n【算法2与真实值的偏差分析】\n")
    print("*所有偏差均按照“真实值相应数据减去预测值相应数据”的方式计算")
    compute_statistical_tests(data_2, data_true)
    if plot:
        # 绘制每一轮测试的得分折线图
        print("绘制每一轮测试得分图...")

        plt.rc('font', family='YouYuan')
        rounds_range = np.arange(1, rounds + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(rounds_range, all_true_scores, marker='o', linestyle='-', color='red', label='真实得分')
        plt.plot(rounds_range, all_scores_1, marker='o', linestyle='-', color='blue', label='算法1 得分')
        plt.plot(rounds_range, all_scores_2, marker='o', linestyle='-', color='green', label='算法2 得分')

        plt.xlabel('测试轮次')
        plt.ylabel('得分')
        plt.title('每一轮测试的得分分布')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("测试完成！")




def genome_scoreonly(A,B):
    score,_=testGeonme_with_given_A_B(A,B)
    return score
def DNN(A,B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_memory("../trained/move_predictor.pth", device)
    move,score=DNNpredict(A,B,model)
    return score
def recursive(A,B):
    score,_=recursive_strategy(A,B)
    return score

def transformer_score(A,B):
    predicted_score=transformer_scoreonly(A, B, "../trained/seq2seq_model.pth")
    return predicted_score

if __name__ == "__main__":


# 显示图表的简易测试rounds=10
#     Compare_TwoModel(genome_scoreonly,transformer_scoreonly,rounds=1000)
    A=[1,2,3,4,5,6]
    B=[1,2,3]

    # print(transformer_scoreonly(A,B))
    Compare_TwoModel(genome_scoreonly,DNN)