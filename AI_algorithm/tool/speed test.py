import time

import torch

from AI_algorithm.Trans import Transformer_predict_batch, TransformerMovePredictor
from AI_algorithm.brute_force import recursive_Strategy
from AI_algorithm.tool.tool import deal_cards_tool

total_time_transformer = 0
total_time_recursive = 0
num_iterations = 20000 # 减少迭代次数以便快速看到结果


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting comparison for {num_iterations} iterations...")
num_a_test = 6  # <--- 修改
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

A_batch=[]
B_batch=[]

model_path_1 = "../trained/transformer_move_predictor_6x3.pth"  # <--- 修改
model1.load_state_dict(torch.load(model_path_1, map_location=device))
for i in range(num_iterations):
    A, B = deal_cards_tool()
    A_batch.append(A)
    B_batch.append(B)

    A_copy = list(A)
    B_copy = list(B)
    start_time_r = time.perf_counter()
    move_r=recursive_Strategy(A_copy, B_copy)
    end_time_r = time.perf_counter()
    total_time_recursive += (end_time_r - start_time_r)



start_time_t = time.perf_counter()
move_t=Transformer_predict_batch(A_batch, B_batch, model1, num_a=num_a_test, num_b=num_b_test)
end_time_t = time.perf_counter()
total_time_transformer += (end_time_t - start_time_t)

#
# # 打开文件并逐行写入
# with open("output.txt", "w", encoding="utf-8") as file:
#     for row in move_t:
#         file.write(" ".join(map(str, row)) + "\n")  # 每行用空格分隔

# --- 报告结果 ---
print("\n--- Comparison Complete ---")
print(f"Total time for Transformer:      {total_time_transformer:.6f} seconds")

print(f"Total time for recursive_Strategy: {total_time_recursive:.6f} seconds")

if num_iterations > 0:
    avg_time_transformer = total_time_transformer / num_iterations
    print(f"\nAverage time per call (Transformer):      {avg_time_transformer:.8f} seconds")

    avg_time_recursive = total_time_recursive / num_iterations
    print(f"Average time per call (recursive_Strategy): {avg_time_recursive:.8f} seconds")
