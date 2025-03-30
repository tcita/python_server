import torch

from AI_algorithm.GA import Get_GA_Strategy

from AI_algorithm.tool.CompareWithOther import DNN_Strategy, TransformerStrategy

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


    score1, _, _, _,A  = simulate_insertion_tool(A, B[strategy[0][0]],  max(1, min(len(A), strategy[0][1])))
    # print(f"1. 将 {B[strategy[0][0]]} 插入A<{strategy[0][1]}>号位  得分: {score1}    A变为{A}")
    score2, _, _, _,A  = simulate_insertion_tool(A, B[strategy[1][0]],  max(1, min(len(A), strategy[1][1])))
    # print(f"2. 将 {B[strategy[1][0]]} 插入A<{strategy[1][1]}>号位  得分: {score2}    A变为{A}")
    score3, _, _, _,A  = simulate_insertion_tool(A, B[strategy[2][0]],  max(1, min(len(A), strategy[2][1])))
    # print(f"3. 将 {B[strategy[2][0]]} 插入A<{strategy[2][1]}>号位  得分: {score3}    A变为{A}")
    # if(strategy[0][1]==strategy[1][1] and  strategy[0][1]!=1):
    #     print(f"A的初始值是{A}    B的初始值是{B}")
    #     # print("有重复的插入位置")
    #     print(strategy)
    return score1+score2+score3
    # print(f"总得分为: {score1 + score2 + score3}")
count = 0
for i in range(100000):
    A,B=deal_cards_tool()
    genome=load_best_genome("../trained/best_genome.pkl")
    GA_Strategy=Get_GA_Strategy(genome,A,B)
    # print("基于遗传算法的启发式方法的仿真模拟")
    ga=strategy_scoring(A,B,GA_Strategy)
    # print("-"*20)
    # print("-"*20)
    # print("深度神经网络的仿真模拟")
    # DNN_Strategy,_=DNNpredict(A,B,"../trained/move_predictor.pth")

    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move=DNN_Strategy(A,B)
    dnn=strategy_scoring(A,B,move)


    TF_Strategy=TransformerStrategy(A,B)
    tr=strategy_scoring(A,B,TF_Strategy)
    # DNNFormatChecking(DNN_Strategy)



    if(tr<dnn):
        count=count+1;
        print(count)
        # print(DNN_Strategy)
# A =[100, 2, 3, 10, 5, 6]  # 示例 A 列表
# B =[6, 7, 8]
# DNN_Strategy,_=DNNpredict(A,B,"../trained/move_predictor.pth")
# dnn=strategy_scoring(A,B,DNN_Strategy)
# print(dnn)
# print(DNN_Strategy)
