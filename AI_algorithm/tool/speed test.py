import time

from AI_algorithm.brute_force import recursive_Strategy
from AI_algorithm.tool.tool import deal_cards_tool
from AI_algorithm.tool.two_model_comparing import Transformer

total_time_transformer = 0
total_time_recursive = 0
num_iterations = 10000 # 减少迭代次数以便快速看到结果，你可以改回 1000000

print(f"Starting comparison for {num_iterations} iterations...")

for i in range(num_iterations):
    # 1. 获取数据
    A, B = deal_cards_tool()

    # --- 计时 Transformer ---
    # 创建副本，以防 Transformer 修改 A, B
    A_copy1 = list(A)
    B_copy1 = list(B)
    start_time_t = time.perf_counter()
    Transformer(A_copy1, B_copy1)
    end_time_t = time.perf_counter()
    total_time_transformer += (end_time_t - start_time_t)

    # --- 计时 recursive_Strategy ---
    # 创建副本，确保它收到的是原始 A, B
    A_copy2 = list(A)
    B_copy2 = list(B)
    start_time_r = time.perf_counter()
    recursive_Strategy(A_copy2, B_copy2)
    end_time_r = time.perf_counter()
    total_time_recursive += (end_time_r - start_time_r)

    # 打印进度 (可选, 对于非常长的循环有用)
    if (i + 1) % (num_iterations // 10) == 0:
        print(f"  ... processed {i+1}/{num_iterations} iterations")

# --- 报告结果 ---
print("\n--- Comparison Complete ---")
print(f"Total time for Transformer:      {total_time_transformer:.6f} seconds")
print(f"Total time for recursive_Strategy: {total_time_recursive:.6f} seconds")

if num_iterations > 0:
    avg_time_transformer = total_time_transformer / num_iterations
    avg_time_recursive = total_time_recursive / num_iterations
    print(f"\nAverage time per call (Transformer):      {avg_time_transformer:.8f} seconds")
    print(f"Average time per call (recursive_Strategy): {avg_time_recursive:.8f} seconds")

# 简单的比较
if total_time_transformer < total_time_recursive:
    print("\nTransformer was faster.")
    if total_time_transformer > 0:
         ratio = total_time_recursive / total_time_transformer
         print(f"recursive_Strategy took approximately {ratio:.2f} times longer.")
elif total_time_recursive < total_time_transformer:
    print("\nrecursive_Strategy was faster.")
    if total_time_recursive > 0:
        ratio = total_time_transformer / total_time_recursive
        print(f"Transformer took approximately {ratio:.2f} times longer.")
else:
    print("\nBoth algorithms took roughly the same amount of time.")