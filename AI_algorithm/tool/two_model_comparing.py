
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
from AI_algorithm.tool.tool import load_best_genome, deal_cards_tool, simulate_insertion_tool




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




def Compare_TwoModel(model, other_model, rounds=1000, plot=False):
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

        # 测试集生成不应该有限制

        A, B = deal_cards_tool()



        print(A)
        print(B)


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

        if(true_score-score_1>=10 ):
            print("算法1得分较低")
            print(A)
            print(B)
            print("算法1这样的移动造成了相对于真实值较低的得分")
            print(f" 算法1的预测移动和预测得分  ：{move_1},{score_1}")
            print(f" 但是真实最佳移动和真实值是：{true_move},{true_score}")


        if(true_score-score_2>=10):
            print("算法2得分较低")
            print(A)
            print(B)
            print("算法2这样的移动造成了相对于真实值较低的得分")
            print(f" 算法2的预测移动和预测得分  ：{move_2},{score_2}")
            print(f" 但是真实最佳移动和真实值是：{true_move},{true_score}")

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
    print("*所有偏差均按照真实值相应数据减去预测值相应数据的方式计算")
    compute_statistical_tests(data_1, data_true)
    print("-" * 20)
    print("\n【算法2与真实值的偏差分析】\n")
    print("*所有偏差均按照真实值相应数据减去预测值相应数据的方式计算")
    compute_statistical_tests(data_2, data_true)

    plot = True  # Set to True to generate plot

    if plot:
        # 创建一个1x2的图表布局
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # 设置全局字体大小
        plt.rcParams.update({'font.size': 12})

        # === 计算全局范围和区间 ===
        interval_width = 5

        # Calculate min/max based on ALL THREE datasets for Y-axis scaling
        overall_min_val = np.min([np.min(data_true), np.min(data_1), np.min(data_2)])
        overall_max_val = np.max([np.max(data_true), np.max(data_1), np.max(data_2)])

        # Define intervals based on this overall range, rounded to interval_width
        min_score = int(np.floor(overall_min_val / interval_width) * interval_width)
        max_score = int(np.ceil(overall_max_val / interval_width) * interval_width)

        # Ensure max_score is at least one interval above min_score
        if max_score <= min_score:
            max_score = min_score + interval_width

        # Recalculate intervals and centers based on the new min_score, max_score
        intervals = list(range(min_score, max_score + interval_width, interval_width))
        if len(intervals) < 2:
            print(f"Warning: Not enough data range ({overall_min_val:.2f} to {overall_max_val:.2f}) "
                  f"to create multiple intervals with width {interval_width}.")
            min_score_adj = int(np.floor(overall_min_val))
            max_score_adj = int(np.ceil(overall_max_val))
            if max_score_adj <= min_score_adj: max_score_adj = min_score_adj + 1
            intervals = [min_score_adj, max_score_adj]
            interval_width_adj = intervals[1] - intervals[0]  # Use adjusted width
            interval_centers = np.array([intervals[0]]) + interval_width_adj / 2.0
            interval_labels = [f"[{intervals[0]},{intervals[1]})"]

        else:
            interval_labels = [f"[{intervals[i]},{intervals[i + 1]})" for i in range(len(intervals) - 1)]
            interval_centers = np.array(intervals[:-1]) + (intervals[1] - intervals[0]) / 2.0

        # === 偏差计算 (基于 data_true 的值落入哪个区间) ===
        interval_errors_1 = [[] for _ in range(len(intervals) - 1)]
        interval_errors_2 = [[] for _ in range(len(intervals) - 1)]

        for i in range(len(data_true)):
            true_val = data_true[i]
            bin_index = -1
            for j in range(len(intervals) - 1):
                if intervals[j] <= true_val < intervals[j + 1]:
                    bin_index = j
                    break
            if bin_index == -1 and true_val == intervals[-1] and len(intervals) > 1:
                bin_index = len(intervals) - 2

            if bin_index != -1 and 0 <= bin_index < len(interval_errors_1):
                interval_errors_1[bin_index].append(data_1[i] - data_true[i])
                interval_errors_2[bin_index].append(data_2[i] - data_true[i])

        mean_errors_1 = [np.mean(errors) if errors else 0 for errors in interval_errors_1]
        mean_errors_2 = [np.mean(errors) if errors else 0 for errors in interval_errors_2]
        sample_counts = [len(errors) for errors in interval_errors_1]

        # === 绘图 ===

        # 1. 小提琴图 (Violin Plot - Left: axs[0])
        violin_parts = axs[0].violinplot([data_true, data_1, data_2], showmeans=True, showmedians=False)
        axs[0].set_title('Score Distribution Violin Plot', fontsize=14)
        axs[0].set_ylabel('Score', fontsize=12)
        axs[0].set_xticks([1, 2, 3])
        axs[0].set_xticklabels(['Ground Truth', 'Algorithm 1', 'Algorithm 2'])
        axs[0].tick_params(axis='both', labelsize=11)
        axs[0].grid(True, linestyle='--', alpha=0.6, axis='y')

        # 设置小提琴图的y轴范围和刻度 (using the overall min/max score)
        axs[0].set_ylim(min_score, max_score)
        axs[0].set_yticks(intervals)
        axs[0].set_yticklabels([f'{i}' for i in intervals])

        # 2. 偏差条形图 (Bias Plot - Right: axs[1]) - Horizontal Bar Chart
        current_interval_width = intervals[1] - intervals[0]  # Use actual width
        bar_total_thickness = current_interval_width * 0.7
        individual_bar_thickness = bar_total_thickness / 2

        # Plot bars
        axs[1].barh(interval_centers - individual_bar_thickness / 2, mean_errors_1,
                    height=individual_bar_thickness, label='Algorithm 1 Bias', color='tab:blue', alpha=0.8)
        axs[1].barh(interval_centers + individual_bar_thickness / 2, mean_errors_2,
                    height=individual_bar_thickness, label='Algorithm 2 Bias', color='tab:green', alpha=0.8)

        # --- Set up axs[1] y-axis to align with axs[0] ---
        axs[1].set_ylim(min_score, max_score)
        axs[1].set_yticks(intervals)
        axs[1].set_yticklabels([f'{i}' for i in intervals])

        axs[1].set_title('Mean Prediction Bias by True Score Interval', fontsize=14)
        axs[1].set_xlabel('Mean Bias (Predicted - True)', fontsize=12)
        axs[1].set_ylabel('True Score', fontsize=12)
        axs[1].tick_params(axis='both', labelsize=11)
        axs[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axs[1].legend(fontsize=10)
        axs[1].grid(True, linestyle='--', alpha=0.6, axis='x')

        # --- 添加样本数量标签 (Adaptive Placement Logic) ---

        # 1. Draw canvas to get reliable initial limits
        fig.canvas.draw()

        # 2. Prepare storage and track needed x-range
        annotations_args = []
        min_x_needed = float('inf')
        max_x_needed = float('-inf')
        xmin_plot_initial, xmax_plot_initial = axs[1].get_xlim()  # Limits after bars are drawn

        # Include bar ends in initial range assessment
        all_bar_ends = [b for b in mean_errors_1 + mean_errors_2 if b is not None and not np.isnan(b)]
        min_bar_end = min(all_bar_ends) if all_bar_ends else xmin_plot_initial
        max_bar_end = max(all_bar_ends) if all_bar_ends else xmax_plot_initial
        min_x_needed = min(min_x_needed, min_bar_end)
        max_x_needed = max(max_x_needed, max_bar_end)

        # 3. Determine annotation positions adaptively
        for i, count in enumerate(sample_counts):
            if count > 0 and i < len(interval_centers):  # Ensure index valid for interval_centers
                bar1_end = mean_errors_1[i]
                bar2_end = mean_errors_2[i]
                abs_bias1 = abs(bar1_end)
                abs_bias2 = abs(bar2_end)
                target_bar_end = bar1_end if abs_bias1 >= abs_bias2 else bar2_end
                is_positive_bar = target_bar_end >= 0

                label_text = f'n={count}'
                plot_width = xmax_plot_initial - xmin_plot_initial
                if plot_width == 0: plot_width = 1
                offset = plot_width * 0.015  # Offset for outside placement
                inner_offset = plot_width * 0.01  # Offset for inside placement

                # Calculate ideal position (outside)
                ideal_x_pos = target_bar_end + offset if is_positive_bar else target_bar_end - offset
                ideal_ha = 'left' if is_positive_bar else 'right'

                # Determine final position
                final_x_pos = ideal_x_pos
                final_ha = ideal_ha
                text_color = 'dimgray'

                # Check boundaries and adjust if needed
                if is_positive_bar and ideal_x_pos > xmax_plot_initial:
                    final_x_pos = target_bar_end - inner_offset
                    final_ha = 'right'
                    text_color = 'white'
                elif not is_positive_bar and ideal_x_pos < xmin_plot_initial:
                    final_x_pos = target_bar_end + inner_offset
                    final_ha = 'left'
                    text_color = 'white'

                # Store args
                annotations_args.append({
                    'text': label_text,
                    'xy': (final_x_pos, interval_centers[i]),
                    'ha': final_ha, 'va': 'center',
                    'fontsize': 8, 'color': text_color,
                    'clip_on': True
                })
                # Update needed range
                min_x_needed = min(min_x_needed, final_x_pos)
                max_x_needed = max(max_x_needed, final_x_pos)

        # 4. Add all annotations
        for args in annotations_args:
            axs[1].annotate(**args)

        # 5. Adjust final X-axis limits with padding
        x_range_needed = max_x_needed - min_x_needed
        if x_range_needed <= 0: x_range_needed = abs(max_x_needed) * 0.1 + 1  # Handle zero/small range
        padding = x_range_needed * 0.05  # 5% padding

        final_xmin = min_x_needed - padding
        final_xmax = max_x_needed + padding
        axs[1].set_xlim(final_xmin, final_xmax)

        # --- 添加偏差图的说明文本 ---
        axs[1].text(0.98, -0.08, "n=sample size in true score interval", transform=axs[1].transAxes,
                    ha='right', va='top', fontsize=9, color='dimgray')

        # --- 添加整体图表标题 ---
        fig.suptitle('Algorithm Performance Comparison: Score Distribution & Prediction Bias', fontsize=16, y=0.99)

        # --- 添加共享的水平参考线 ---
        for boundary in intervals:
            # Use slightly different alpha/style for less visual clutter if desired
            axs[0].axhline(y=boundary, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
            axs[1].axhline(y=boundary, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)

        # --- 调整布局并显示 ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust rect for titles/labels
        plt.show()
  

    print("测试完成！")



def genome(A,B):
    best_genome = load_best_genome()
    move=GA_Strategy(best_genome, A, B)
    return move


def DNN(A,B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_memory("../trained/move_predictor.pth", device)
    move,_=DNNpredict(A,B,model)
    return move


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

i = 0
def Transformer(A, B):
    # 加入专门预测低分的模型
    global i

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



    score1=strategy_TrueScore(A,B,move1)

    genome=load_best_genome("../trained/best_genome.pkl")
    move2=GA_Strategy(genome,A, B)

    score2=strategy_TrueScore(A,B,move2)
    if score1<20:
        print(f"异常值出现 {A},{B} {move1} ")

    if(score1<score2 ):
        i+=1
        print(f"assist win {A},{B} {i} times")

        print(f"Transformer得分：{score1}")
        print(f"assist得分：{score2}")
        return move2
    else:
        return move1

    # return move1



def recursive(A,B):
    score,strategy=recursive_StrategyAndScore(A, B)
    return score,strategy



if __name__ == "__main__":



    Compare_TwoModel(genome,Transformer,rounds=10000,plot=True)
