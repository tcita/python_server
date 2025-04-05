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
from AI_algorithm.tool.tool import deal_cards_tool, load_best_genome, simulate_insertion_tool, complete_best_moves


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

def analyze_model_errors(model_strategy, threshold=10, num_samples=1000):
    """
    分析Transformer模型预测得分与真实得分差距较大的情况，并将结果保存为指定格式的JSON文件
    """
    large_gap_cases = []
    formatted_cases = []  # 用于存储按指定格式整理的案例
    
    for i in range(num_samples):
        # 生成随机的A和B
        A, B = deal_cards_tool()

        # 获取模型预测的策略
        if(model_strategy==GA_Strategy):
            genome = load_best_genome()
            predicted_strategy = model_strategy(genome,A, B)
        else:
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

        # 保存错误案例
        # # 保存格式化的错误案例到JSON文件
        # import json
        #
        # if model_strategy==Transformer:
        #     filename = "../json/transformer_error_cases.json"
        # if model_strategy==GA_Strategy:
        #     filename = "../json/GA_error_cases.json"
        #
        # with open(filename, 'w') as f:
        #     json.dump(formatted_cases, f, indent=4)
        # print("\n错误案例已按指定格式保存到 transformer_error_cases.json")
    
    return large_gap_cases



if __name__ == '__main__':


    analyze_model_errors(Transformer)
    # analyze_model_errors(recursive_Strategy)
    # analyze_model_errors(GA_Strategy)



