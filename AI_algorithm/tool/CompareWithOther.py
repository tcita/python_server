import io
import pickle
import time
import numpy as np
import torch
from scipy import stats


from sympy import false

from AI_algorithm.Deep_Neural_Network import  DNNpredict
from AI_algorithm.GA import  GA_Strategy

import matplotlib.pyplot as plt

from AI_algorithm.Trans import TransformerMovePredictor, Transformer_predict
from AI_algorithm.Trans_assist import TransformerMovePredictor_assist, Transformer_predict_assist
from AI_algorithm.brute_force import recursive_StrategyAndScore
from AI_algorithm.server import load_model_from_memory
from AI_algorithm.tool.tool import load_best_genome, deal_cards_tool, simulate_insertion_tool, complete_best_moves

genome_loaded = load_best_genome()


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

# def  testGeonme_with_given_A_B(A, B, best_genome=genome_loaded):
#
#
#     # 复制 A 避免修改原始数据
#     genetic_A = A.copy()
#     genetic_score = 0
#
#     # 计算遗传算法的运行时间
#     start_time_genetic = time.time()
#     for i, card in enumerate(B):
#         remaining_B = B[i + 1 :]
#         _, score, genetic_A = genome_choose_insertion(best_genome, genetic_A, card, remaining_B)
#         genetic_score += score
#     end_time_genetic = time.time()
#     time_genetic = end_time_genetic - start_time_genetic  # 计算遗传算法时间
#
#     # # 计算 other_model 算法的运行时间
#     # start_time_other = time.time()
#     # other_score = other_model(A.copy(), B.copy())
#     # end_time_other = time.time()
#     # time_other = end_time_other - start_time_other  # 计算其他算法时间
#
#     return genetic_score, time_genetic
def TestModel_with_given_A_B(other_model, A, B):
    # 计算 other_model 算法的运行时间
    start_time_other = time.time()
    move = other_model(A.copy(), B.copy())
    score=strategy_TrueScore(A, B, move)

    end_time_other = time.time()
    time_other = end_time_other - start_time_other  # 计算其他算法时间
    return score, time_other,move
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

            # 检查 A 和 B 是否有任何重复的元素,确保只用能得分的A,B训练
            if  set(A) & set(B):  # 如果 A 和 B 有重复元素
                break  # 退出循环，继续处理这对 A, B
        print("  生成 A, B 成功")
        print(A)
        print(B)

# 这里的move并没有使用 计算它是为了统计时间
        score_1, time_1,move_1 = TestModel_with_given_A_B(model, A, B)


        times_list1.append(time_1)


        score_2, time_2 ,move_2= TestModel_with_given_A_B(other_model, A, B)

        times_list2.append(time_2)

        true_score, true_move = recursive(A,B)
        print('-'*20)
        print(f" 真实值 ：{true_score}")
        print(f" 算法1得分 ：{score_1}")
        print(f" 算法2得分 ：{score_2}")
        print('-' * 20)
        # 检查差距过大的策略
        print("这样的移动造成了相对于真实值较低的得分")
        if(true_score-score_1>=10 ):
            print("算法1得分较低")
            print(A)
            print(B)
            print(f" 算法1  ：{move_1}")
            print(f" 真实最佳移动：{true_move}")
            print(f" 真实值：{true_score}")
            print(f" 算法1 得分：{score_1}")


        if(true_score-score_2>=10):
            print("算法2得分较低")
            print(A)
            print(B)
            print(f" 算法2  ：{move_2}")
            print(f" 真实最佳移动：{true_move}")
            print(f" 真实值：{true_score}")
            print(f" 算法2 得分：{score_2}")
            print('-'*20)

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

        # 绘制散点图
        plt.scatter(rounds_range, all_true_scores, color='red', label='真实得分')  # 使用 scatter 绘制散点
        plt.scatter(rounds_range, all_scores_1, color='blue', label='算法1 得分')
        plt.scatter(rounds_range, all_scores_2, color='green', label='算法2 得分')

        plt.xlabel('测试轮次')
        plt.ylabel('得分')
        plt.title('每一轮测试的得分分布')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("测试完成！")



def genome(A,B):
    best_genome = genome_loaded
    move=GA_Strategy(best_genome, A, B)
    return move


def DNN(A,B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_memory("../trained/move_predictor.pth", device)
    move,_=DNNpredict(A,B,model)
    return move


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Transformer(A, B):
    # 加入专门预测低分的模型
    num_a_test = 6 # <--- 修改
    num_b_test = 3
    # 确保这些参数与训练时一致
    d_model = 256
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    model1 = TransformerMovePredictor(
        num_a=num_a_test, num_b=num_b_test, d_model=d_model,
        nhead=nhead, num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward
    ).to(device)



    model_path_1 = "../trained/transformer_move_predictor_6x3.pth" # <--- 修改
    model1.load_state_dict(torch.load(model_path_1, map_location=device))
    move1, _= Transformer_predict(A, B, model1, num_a=num_a_test, num_b=num_b_test)

    # 专门预测低分的模型

    score1=strategy_TrueScore(A,B,move1)
    move2=DNN(A,B)
    move2=complete_best_moves(move2)
    score2=strategy_TrueScore(A,B,move2)

    #
    # move2=GA_Strategy(A, B)
    # move2=complete_best_moves(move2)
    # score2=strategy_TrueScore(A,B,move2)

    if(score1<score2 ):
        print(f"assist win {A},{B}")

        print(f"Transformer得分：{score1}")
        print(f"assist得分：{score2}")
        return move2
    else:
        return move1
    return move1

    #
    # if score2 > score1:
    #
    #     move3, _ = Transformer_predict_assist(A, B, model3, num_a=num_a_test, num_b=num_b_test)
    #
    #
    #     score3 = strategy_TrueScore(A, B, move3)
    #     if score3 > score2:
    #         print(f"HI move3")
    #         print(f"普通Transformer得分：{score1}")
    #         print(f"GA得分：{score2}")
    #         print(f"辅助Transformer得分：{score3}")
    #         return move3
    #     else:
    #         print(f"HI move2")
    #         print(f"普通Transformer得分：{score1}")
    #         print(f"GA得分：{score2}")
    #         print(f"辅助Transformer得分：{score3}")
    #         return move2
    # else:



def recursive(A,B):
    score,strategy=recursive_StrategyAndScore(A, B)
    return score,strategy

# 这个预测不了策略
# def transformer_score(A,B):
#     predicted_score=transformer_scoreonly(A, B, "../trained/seq2seq_model.pth")
#     return predicted_score

if __name__ == "__main__":



    Compare_TwoModel(Transformer,Transformer,rounds=1000,plot=False)
    # A=[7, 9, 5, 13, 3, 10]
    # B=[7, 5, 5]
    # A=[11, 5, 13, 10, 1, 12]
    # B=[1, 9, 4]
    # s=[[1, 0], [2, 0], [0, 0]]
    # print(strategy_TrueScore(A,B,s))
#     print(DNN(A,B))
#     Compare_TwoModel(Transformerscore,Transformerscore)