import os

from scipy.stats import shapiro, anderson, probplot, skew, kurtosis

from collections import Counter


from AI_algorithm.GA import Get_GA_Strategy
from AI_algorithm.brute_force import recursive_strategy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
def Score_distribution():
    re=[]
    for i in range(200000):
        A, B = deal_cards_tool()
        max_score, best_moves = recursive_strategy(A, B)

        # 提取已有的第一个元素
        existing_first_elements = {move[0] for move in best_moves}

        # 需要填充的第一个元素（确保 0, 1, 2 都存在）
        missing_first_elements = [x for x in [0, 1, 2] if x not in existing_first_elements]

        # 填充 best_moves 至少 3 个子数组
        while len(best_moves) < 3:
            first_element = missing_first_elements.pop(0)  # 获取缺失的第一个元素
            best_moves.append([first_element, 0])  # 组合 [缺失的元素, 0]

        true_max_score = strategy_scoring(A, B, best_moves)
        if (max_score != true_max_score):
            raise ValueError(f"递归计算的 max_score ({max_score}) 出错   ({A},{B},{best_moves})")
        print(f"{i} of {200000}")
        re.append(max_score)
    return re


if __name__ == '__main__':
    # 统计每个输出值的频率
    # 定义文件名
    file_name = 'Score_Distribution.npy'

    # 检查文件是否存在
    if os.path.exists(file_name):
        # 如果文件存在，加载数据
        data_raw = np.load(file_name)
        print("数据已加载")
    else:
        # 如果文件不存在，创建数据并保存
        data_raw=Score_distribution()  # 你可以根据需要修改这行数据生成逻辑
        np.save(file_name, data_raw)
        print("数据已创建并保存")

    frequency = Counter(data_raw)

    # 打印前 10 个最常见的结果及其频率
    print("最常见的 10 个结果及其频率：")
    for value, count in frequency.most_common(10):
        print(f"值: {value}, 频率: {count}")

    # # 可视化分布
    # values, counts = zip(*frequency.items())
    # plt.figure(figsize=(10, 6))
    # plt.bar(values, counts, color='skyblue', edgecolor='black')
    # plt.title("Score Distribution")
    # plt.xlabel("Score")
    # plt.ylabel("Frequency")
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    #

    import math  # 用于获取 pi

    # 将列表转换为 NumPy 数组
    data = np.array(data_raw)
    print(f"数据点数量: {len(data)}")

    # --- 2. 计算均值 (μ) 和标准差 (σ) ---
    # 对于正态分布，样本均值和样本标准差是其参数的最佳估计
    mu_fit = np.mean(data)
    sigma_fit = np.std(data)  # 注意：np.std 默认计算总体标准差 (ddof=0), 这通常用于拟合

    print(f"拟合得到的均值 (μ_fit): {mu_fit:.4f}")
    print(f"拟合得到的标准差 (σ_fit): {sigma_fit:.4f}")

    # --- 3. 得到正态分布的数学表达式 (概率密度函数 PDF) ---
    # 正态分布 PDF 公式: f(x | μ, σ) = (1 / (σ * sqrt(2π))) * exp(-(x - μ)² / (2 * σ²))

    # 构建包含拟合参数的表达式字符串
    expression = f"f(x) = (1 / ({sigma_fit:.4f} * sqrt(2*π))) * exp(-(x - {mu_fit:.4f})**2 / (2 * {sigma_fit:.4f}**2))"
    print("\n拟合的正态分布概率密度函数 (PDF) 表达式:")
    print(expression)

    # --- 4. (可选) 使用 SciPy 的 norm 对象进行计算和可视化 ---

    # 创建一个代表拟合后正态分布的 SciPy 对象
    fitted_distribution = norm(loc=mu_fit, scale=sigma_fit)

    # 你可以使用这个对象来计算特定点的概率密度值(PDF)或累积分布函数(CDF)等
    x_value = 110
    pdf_at_x = fitted_distribution.pdf(x_value)
    cdf_at_x = fitted_distribution.cdf(x_value)
    print(f"\n使用 SciPy 计算:")
    print(f"在 x = {x_value} 处的概率密度值 (PDF): {pdf_at_x:.6f}")
    print(f"x <= {x_value} 的累积概率 (CDF): {cdf_at_x:.6f}")

    # 可视化拟合效果
    print("\n正在生成可视化图表...")
    plt.figure(figsize=(12, 7))

    # 绘制数据的直方图 (归一化，使其面积为 1)
    # 可以调整 bins 的数量来改变直方图的精细度
    plt.hist(data, bins=18, density=True, alpha=0.7, color='skyblue', label='数据直方图 (归一化)')

    # 绘制拟合的正态分布 PDF 曲线
    xmin, xmax = plt.xlim()  # 获取直方图的 x 轴范围
    x_plot = np.linspace(xmin, xmax, 200)  # 在此范围内生成一系列 x 值
    pdf_plot = fitted_distribution.pdf(x_plot)  # 计算这些 x 值对应的 PDF 值
    plt.plot(x_plot, pdf_plot, 'r-', lw=2, label='拟合的正态分布 PDF 曲线')

    # 添加图例、标题和标签
    plt.title(f'数据分布与拟合的正态分布\n拟合参数: μ = {mu_fit:.4f}, σ = {sigma_fit:.4f}')
    plt.xlabel('数值')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("可视化图表显示完毕.")


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
