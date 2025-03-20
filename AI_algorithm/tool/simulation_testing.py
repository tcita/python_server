from AI_algorithm.Deep_Neural_Network import DNNpredict
from AI_algorithm.GA import Get_GA_Strategy
from AI_algorithm.tool.tool import deal_cards_tool, load_best_genome, simulate_insertion_tool


def strategy_scoring(A, B, strategy):

    print(f"A的初始值是{A}    B的初始值是{B}")
    score1, _, _, _,A  = simulate_insertion_tool(A, B[strategy[0][0]],  max(1, min(len(A), strategy[0][1])))
    print(f"1. 将 {B[strategy[0][0]]} 插入A<{strategy[0][1]}>号位  得分: {score1}    A变为{A}")
    score2, _, _, _,A  = simulate_insertion_tool(A, B[strategy[1][0]],  max(1, min(len(A), strategy[1][1])))
    print(f"2. 将 {B[strategy[1][0]]} 插入A<{strategy[1][1]}>号位  得分: {score2}    A变为{A}")
    score3, _, _, _,A  = simulate_insertion_tool(A, B[strategy[2][0]],  max(1, min(len(A), strategy[2][1])))
    print(f"3. 将 {B[strategy[2][0]]} 插入A<{strategy[2][1]}>号位  得分: {score3}    A变为{A}")
    print(f"总得分为: {score1 + score2 + score3}")

A,B=deal_cards_tool()
genome=load_best_genome("../trained/best_genome.pkl")
GA_Strategy=Get_GA_Strategy(genome,A,B)
print("基于遗传算法的启发式方法的仿真模拟")
strategy_scoring(A,B,GA_Strategy)
print("-"*20)
print("-"*20)
print("深度神经网络的仿真模拟")
DNN_Strategy,_=DNNpredict(A,B,"../trained/move_predictor.pth")
strategy_scoring(A,B,DNN_Strategy)
print("-"*20)
print("-"*20)

