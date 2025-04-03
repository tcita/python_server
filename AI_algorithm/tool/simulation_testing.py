import os

from scipy.stats import shapiro, anderson, probplot, skew, kurtosis

from collections import Counter
from scipy.stats import gaussian_kde
import seaborn as sns
from AI_algorithm.GA import GA_Strategy
from AI_algorithm.brute_force import recursive_StrategyAndScore, recursive_Strategy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from AI_algorithm.tool.CompareWithOther import Transformer
from AI_algorithm.tool.tool import deal_cards_tool, load_best_genome, simulate_insertion_tool



def DNNFormatChecking(DNNStrategy):
    # 用于存储第一个值
    first_values = []

    # 遍历二维数组的每一行
    for row in DNNStrategy:
        first_value = row[0]

        # 检查第一个值是否在 0, 1, 2 中，并且不重复
        if first_value in {0, 1, 2}:
            if first_value in first_values:
                raise ValueError(f"第一个值 {first_value} 重复，程序终止！")
            first_values.append(first_value)
        else:
            raise ValueError(f"第一个值 {first_value} 不在 0, 1, 2 之内，程序终止！")


def strategy_scoring(A, B, strategy):

    # min(len(A), strategy[0][1])是为了处理插入位置越界的情况
    score1, _, _, _,A  = simulate_insertion_tool(A, B[strategy[0][0]],   min(len(A), strategy[0][1]))
    # print(f"1. 将 {B[strategy[0][0]]} 插入A<{strategy[0][1]}>号位  得分: {score1}    A变为{A}")
    score2, _, _, _,A  = simulate_insertion_tool(A, B[strategy[1][0]],   min(len(A), strategy[1][1]))
    # print(f"2. 将 {B[strategy[1][0]]} 插入A<{strategy[1][1]}>号位  得分: {score2}    A变为{A}")
    score3, _, _, _,A  = simulate_insertion_tool(A, B[strategy[2][0]],   min(len(A), strategy[2][1]))
    # print(f"3. 将 {B[strategy[2][0]]} 插入A<{strategy[2][1]}>号位  得分: {score3}    A变为{A}")
    # if(strategy[0][1]==strategy[1][1] and  strategy[0][1]!=1):
    #     print(f"A的初始值是{A}    B的初始值是{B}")
    #     # print("有重复的插入位置")
    #     print(strategy)
    return score1+score2+score3
    # print(f"总得分为: {score1 + score2 + score3}")

def analyze_model_errors(model_strategy, threshold=5, num_samples=200000):
    """
    分析Transformer模型预测得分与真实得分差距较大的情况，并将结果保存为指定格式的JSON文件
    """
    large_gap_cases = []
    formatted_cases = []  # 用于存储按指定格式整理的案例
    
    for i in range(num_samples):
        # 生成随机的A和B
        A, B = deal_cards_tool()

        # 获取模型预测的策略

        predicted_strategy = model_strategy(A, B)
        # 填充策略
        existing_first_elements = {move[0] for move in predicted_strategy}
        missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]
        while len(predicted_strategy) < 3:
            first_element = missing_first_elements.pop(0)
            predicted_strategy.append([first_element, 0])

        pred_score = strategy_scoring(A.copy(), B, predicted_strategy)
        # 计算真实的最优策略和得分
        true_strategy = recursive_Strategy(A, B)

        # 填充真实值策略
        existing_first_elements = {move[0] for move in true_strategy}
        missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]
        while len(true_strategy) < 3:
            first_element = missing_first_elements.pop(0)
            true_strategy.append([first_element, 0])

        true_score = strategy_scoring(A.copy(), B, true_strategy)
        
        # 计算预测得分与真实得分的差距
        score_gap = abs(true_score - pred_score)
        
        if score_gap >= threshold:
            # 保存详细信息用于控制台输出
            case_info = {
                'A': A,
                'B': B,
                'true_strategy': true_strategy,
                'predicted_strategy': predicted_strategy,
                'true_score': true_score,
                'predicted_score': pred_score,
                'score_gap': score_gap
            }
            large_gap_cases.append(case_info)
            
            # 按指定格式保存案例
            formatted_case = {
                'A': A,
                'B': B,
                'max_score': true_score,  # 使用true_score作为max_score
                'best_moves': true_strategy  # 使用true_strategy作为best_moves
            }
            formatted_cases.append(formatted_case)
            
            print(f"\n发现差距较大的案例 ({i+1}/{num_samples}):")
            print(f"A: {A}")
            print(f"B: {B}")
            print(f"真实策略: {true_strategy}")
            print(f"预测策略: {predicted_strategy}")
            print(f"真实得分: {true_score}")
            print(f"预测得分: {pred_score}")
            print(f"得分差距: {score_gap}")
            print("-" * 50)
    
    # 输出统计信息
    total_error_cases = len(large_gap_cases)
    print(f"\n分析总结:")
    print(f"分析样本总数: {num_samples}")
    print(f"发现差距大于{threshold}分的案例数: {total_error_cases}")
    print(f"错误率: {(total_error_cases/num_samples)*100:.2f}%")
    
    if large_gap_cases:
        # 计算平均差距
        avg_gap = sum(case['score_gap'] for case in large_gap_cases) / len(large_gap_cases)
        print(f"平均得分差距: {avg_gap:.2f}")
        
        # 找出差距最大的案例
        max_gap_case = max(large_gap_cases, key=lambda x: x['score_gap'])
        print(f"\n差距最大的案例:")
        print(f"A: {max_gap_case['A']}")
        print(f"B: {max_gap_case['B']}")
        print(f"真实策略: {max_gap_case['true_strategy']}")
        print(f"预测策略: {max_gap_case['predicted_strategy']}")
        print(f"真实得分: {max_gap_case['true_score']}")
        print(f"预测得分: {max_gap_case['predicted_score']}")
        print(f"得分差距: {max_gap_case['score_gap']}")
        
        # 保存格式化的错误案例到JSON文件
        import json

        if model_strategy==Transformer:
            filename = "../json/transformer_error_cases.json"
        if model_strategy==GA_Strategy:
            filename = "../json/GA_error_cases.json"

        with open(filename, 'w') as f:
            json.dump(formatted_cases, f, indent=4)
        print("\n错误案例已按指定格式保存到 transformer_error_cases.json")
    
    return large_gap_cases

def Score_distribution(model_strategy):
    re = []
    error_cases = []  # 新增：用于存储差距大的案例
    
    for i in range(10000):
        A, B = deal_cards_tool()

        best_moves = model_strategy(A, B)

        # 填充策略
        # 提取已有的第一个元素
        existing_first_elements = {move[0] for move in best_moves}

        # 需要填充的第一个元素（确保 0, 1, 2 都存在）
        missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]

        # 填充 best_moves 至少 3 个子数组
        while len(best_moves) < 3:
            first_element = missing_first_elements.pop(0)
            best_moves.append([first_element, 0])

        # 如果是Transformer模型，还要计算与真实得分的差距
        if model_strategy == Transformer:
            pred_score = strategy_scoring(A.copy(), B, best_moves)
            true_strategy = recursive_Strategy(A, B)
            true_score = strategy_scoring(A.copy(), B, true_strategy)
            
            # 检查得分差距
            score_gap = abs(true_score - pred_score)
            if score_gap >= 5:  # 差距阈值设为5
                error_case = {
                    'A': A,
                    'B': B,
                    'true_strategy': true_strategy,
                    'predicted_strategy': best_moves,
                    'true_score': true_score,
                    'predicted_score': pred_score,
                    'score_gap': score_gap
                }
                error_cases.append(error_case)
                print(f"\n发现差距较大的案例 ({i+1}/10000):")
                print(f"A: {A}")
                print(f"B: {B}")
                print(f"真实策略: {true_strategy}")
                print(f"预测策略: {best_moves}")
                print(f"真实得分: {true_score}")
                print(f"预测得分: {pred_score}")
                print(f"得分差距: {score_gap}")
                print("-" * 50)
        
        true_max_score = strategy_scoring(A, B, best_moves)
        print(f"{i} of {10000}")
        re.append(true_max_score)
    
    # 如果是Transformer模型，输出错误分析结果
    if model_strategy == Transformer and error_cases:
        total_error_cases = len(error_cases)
        print(f"\n分析总结:")
        print(f"分析样本总数: 10000")
        print(f"发现差距大于5分的案例数: {total_error_cases}")
        print(f"错误率: {(total_error_cases/10000)*100:.2f}%")
        
        # 计算平均差距
        avg_gap = sum(case['score_gap'] for case in error_cases) / len(error_cases)
        print(f"平均得分差距: {avg_gap:.2f}")
        
        # 找出差距最大的案例
        max_gap_case = max(error_cases, key=lambda x: x['score_gap'])
        print(f"\n差距最大的案例:")
        print(f"A: {max_gap_case['A']}")
        print(f"B: {max_gap_case['B']}")
        print(f"真实策略: {max_gap_case['true_strategy']}")
        print(f"预测策略: {max_gap_case['predicted_strategy']}")
        print(f"真实得分: {max_gap_case['true_score']}")
        print(f"预测得分: {max_gap_case['predicted_score']}")
        print(f"得分差距: {max_gap_case['score_gap']}")
        
        # 保存错误案例到文件
        import json
        with open('transformer_prediction_errors.json', 'w') as f:
            json.dump(error_cases, f, indent=2)
        print("\n错误案例已保存到 transformer_prediction_errors.json")
    
    return re


if __name__ == '__main__':
    genome=load_best_genome("../trained/best_genome.pkl")



    # 真实分布
    file_name0 = 'Score_Distribution_10k.npy'
    model0=recursive_Strategy


    # # GA预测分布
    # file_name1 = 'Score_Distribution_GA_10k.npy'
    # model1=GA_Strategy


    # transformer预测分布
    file_name1 = 'Score_Distribution_Transformer_10k.npy'
    model1=Transformer


    # 检查文件是否存在
    if os.path.exists(file_name1):
        # 如果文件存在，加载数据
        data1 = np.load(file_name1)
        print("数据已加载")
    else:
        # 如果文件不存在，创建数据并保存
        data1=Score_distribution(model1)
        np.save(file_name1, data1)
        print("数据已创建并保存")

    # 检查文件是否存在
    if os.path.exists(file_name0):
        # 如果文件存在，加载数据
        data0 = np.load(file_name0)
        print("数据已加载")
    else:
        # 如果文件不存在，创建数据并保存
        data0 = Score_distribution(model0)
        np.save(file_name0, data0)
        print("数据已创建并保存")

    # frequency = Counter(data_raw)
    #
    # # 打印前 10 个最常见的结果及其频率
    # print("最常见的 10 个结果及其频率：")
    # for value, count in frequency.most_common(10):
    #     print(f"值: {value}, 频率: {count}")

    # 可视化分布
    # 绘制直方图
    analyze_model_errors(model1)
    if 1==0:
        plt.figure(figsize=(10, 6))

        # 绘制第一组数据的直方图
        bin = 19
        plt.hist(data1, bins=bin, alpha=0.7, color='skyblue', label='Predicted Value By Model 1')

        # 绘制第二组数据的直方图
        plt.hist(data0, bins=bin, alpha=0.5, color='orange', label='Ground Truth')

        # 添加核密度估计（KDE）曲线

        # 对 Data 1 进行 KDE
        kde_data1 = gaussian_kde(data1)
        x_vals = np.linspace(min(data1), max(data1), 1000)  # 生成平滑的 x 值
        # 计算缩放因子
        bin_width = (max(data1) - min(data1)) / bin
        scaling_factor = len(data1) * bin_width
        plt.plot(x_vals, kde_data1(x_vals) * scaling_factor, color='navy', label='KDE Predicted Value By Model 1')

        # 对 Data 0 进行 KDE
        kde_data0 = gaussian_kde(data0)
        bin_width = (max(data0) - min(data0)) / bin
        scaling_factor = len(data0) * bin_width
        plt.plot(x_vals, kde_data0(x_vals) * scaling_factor, color='darkorange', label='KDE Ground Truth')

        # 添加标题和坐标轴标签
        plt.title("Score Distribution with KDE")
        plt.xlabel("Score")
        plt.ylabel("Frequency")

        # 添加图例
        plt.legend()

        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 显示图表
        plt.show()




# 备用函数

# count = 0
# for i in range(100000):
#     A,B=deal_cards_tool()
    # genome=load_best_genome("../trained/best_genome.pkl")
    # GA_Strategy=Get_GA_Strategy(genome,A,B)
    # # print("基于遗传算法的启发式方法的仿真模拟")
    # ga=strategy_scoring(A,B,GA_Strategy)
    # # print("-"*20)
    # # print("-"*20)
    # # print("深度神经网络的仿真模拟")
    # # DNN_Strategy,_=DNNpredict(A,B,"../trained/move_predictor.pth")
    #
    # #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _,move=DNN(A,B)
    # dnn=strategy_scoring(A,B,move)
    #
    #
    # _,TF_Strategy=Transformer(A,B)
    # tr=strategy_scoring(A,B,TF_Strategy)
    # DNNFormatChecking(DNN_Strategy)



    #
    # if(tr<dnn):
    #     count=count+1;
    #     print(count)
        # print(DNN_Strategy)
# A =[100, 2, 3, 10, 5, 6]  # 示例 A 列表
# B =[6, 7, 8]
# DNN_Strategy,_=DNNpredict(A,B,"../trained/move_predictor.pth")
# dnn=strategy_scoring(A,B,DNN_Strategy)
# print(dnn)
# print(DNN_Strategy)
