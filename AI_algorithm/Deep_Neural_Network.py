import time

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from AI_algorithm.GA import simulate_insertion
from AI_algorithm.brute_force import recursive_strategy
from AI_algorithm.tool.tool import deal_cards_tool, calculate_score_by_strategy

jsonfilename="json/data_raw.json"
GPUDEBUG_MODE = False  # 开启调试模式时设为True，关闭时设为False

def conditional_print(*args, **kwargs):
    if GPUDEBUG_MODE:
        print(*args, **kwargs)


# 检查CUDA可用性并强制使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用，使用CPU")


def calculate_score(A, B, order, moves):
    """
    根据给定的插入顺序和位置计算总得分
    """


    strategy = [order[0],moves[0]],[order[1],moves[1]],[order[2],moves[2]]

    # ordered_B = [B[i] for i in order]
    # for pos in moves:
    #     score, _, _, _, new_A = simulate_insertion(current_A, ordered_B.pop(0), pos)
    #     total_score += score
    #     current_A = new_A
    total_score=calculate_score_by_strategy(A, B, strategy)
    return total_score


class MovePredictor(nn.Module):
    def __init__(self, input_size: int):
        super(MovePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # 添加批归一化
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)   # 添加批归一化
        self.fc3 = nn.Linear(64, 32)    # 额外增加一层隐藏层
        self.bn3 = nn.BatchNorm1d(32)   # 添加批归一化
        self.order_head = nn.Linear(32, 9)  # 输出展平后为3*3
        self.pos_head = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # 归一化后 ReLU
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))  # 新增隐藏层
        order_logits = self.order_head(x).view(-1, 3, 3)
        pos_preds = self.pos_head(x)
        return order_logits, pos_preds



def sample_order(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    order = torch.argsort(-probs, dim=-1)  # 贪心选择概率最高的顺序
    return order[:, :, 0].squeeze()


def prepare_data(sample: dict):
    """
    数据预处理：
      - 检查A和B是否存在交集（过滤噪音样本）
      - 构建输入向量以及目标输出（顺序与位置）
    """
    A = sample["A"]
    B = sample["B"]
    # best_moves = sample["best_moves"]


    # 打印当前处理的 A, B 和 best_moves
    # print(f"正在处理的样本:")
    # print(f"A: {A}")
    # print(f"B: {B}")
    # print(f"Best Moves: {best_moves}")
    # print(f"Best Moves: {best_moves}")

    best_moves = sample["best_moves"]
    # 将A和B拼接成输入向量（假设长度为9）
    input_vector = np.array(A + B, dtype=np.float32)
    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)
    return input_vector, order_target, pos_target


def train_model(train_data, epochs=2000, batch_size=512, model_path="./trained/move_predictor.pth"):
    input_size = 9
    model = MovePredictor(input_size).to(device)
    conditional_print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    conditional_print("模型是否在GPU上:", next(model.parameters()).is_cuda)  # 检查模型是否在GPU
    conditional_print("显存已分配总量:", torch.cuda.memory_allocated() / 1e6, "MB")  # 当前显存分配
    conditional_print("显存峰值:", torch.cuda.max_memory_allocated() / 1e6, "MB")  # 历史最大显存使用

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)
    # pos_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            inputs, order_targets, pos_targets = [], [], []

            for sample in batch:
                input_vec, order_tgt, pos_tgt = prepare_data(sample)
                if input_vec is None:  # 跳过噪音样本
                    continue
                inputs.append(input_vec)
                order_targets.append(order_tgt)
                pos_targets.append(pos_tgt)
            #
            # # 打印调试信息
            # print("Order Targets Shapes:", [len(ot) for ot in order_targets])
            # print("Position Targets Shapes:", [len(pt) for pt in pos_targets])

            if not inputs:
                continue

            start_time = time.perf_counter()
            inputs = torch.FloatTensor(np.array(inputs)).to(device)
            order_targets = torch.LongTensor(np.array(order_targets)).to(device)

            pos_targets = [pos_tgt - 1 for pos_tgt in pos_targets]  # 变成 0 到 len(A)-1
            pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device)

            end_time = time.perf_counter()

            conditional_print(f"数据加载时间: {end_time - start_time:.4f} 秒")
            conditional_print(f"Batch输入加载后显存: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

            # 前向传播前
            conditional_print("前向传播前显存:", torch.cuda.memory_allocated() / 1e6, "MB")
            order_logits, pos_preds = model(inputs)
            conditional_print("前向传播后显存:", torch.cuda.memory_allocated() / 1e6, "MB")
            # 将order_logits调整为 (batch_size*3, 3)，order_targets为 (batch_size*3)
            order_loss = order_criterion(order_logits.view(-1, 3), order_targets.view(-1))
            pos_loss = pos_criterion(pos_preds, pos_targets)
            loss = order_loss + pos_loss

            # 反向传播前
            conditional_print("反向传播前显存:", torch.cuda.memory_allocated() / 1e6, "MB")
            optimizer.zero_grad()
            loss.backward()

            # 反向传播后
            conditional_print("反向传播后显存:", torch.cuda.memory_allocated() / 1e6, "MB")
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if epoch % 100 == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")
    return model


def DNNpredict(A, B, model_path="./trained/move_predictor.pth"):
    """
    根据输入的A和B利用训练好的神经网络模型预测最佳移动和得分
    """
    try:
        # 如果A和B没有交集且检查B中有重复元素，则不需要预测  返回固定的移动和得分0 如  1， [2,3,4,5,6,7] ，1
        if not set(A) & set(B) and len(B) == len(set(B)):
            return [[0, 1], [1, 1], [2, 1]], 0

        input_size = 9
        model = MovePredictor(input_size).to(device)

        try:

            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"DNN Model successfully loaded from {model_path}")
        except FileNotFoundError:
            # 文件不存在时的处理
            print(f"Error: The file '{model_path}' was not found. Please check the file path.")
        except PermissionError:
            # 权限不足时的处理
            print(f"Error: Permission denied when accessing the file '{model_path}'.")
        except RuntimeError as e:
            # 模型权重文件损坏或与模型结构不匹配时的处理
            if "state_dict" in str(e):
                print(f"Error: The file '{model_path}' might be corrupted or incompatible with the model architecture.")
            else:
                print(f"Runtime error occurred while loading the model weights: {e}")
        except Exception as e:
            # 捕获其他未知异常
            print(f"An unexpected error occurred while loading the model weights from '{model_path}': {e}")
        model.eval()

        input_vector = np.array(A + B, dtype=np.float32)
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        order = sample_order(order_logits).squeeze(0).cpu().numpy()
        # # 如果采样结果无效，则采用默认顺序
        # if len(set(order)) != 3:
        #     order = np.array([0, 1, 2])

        pos_range = len(A)
        pos_preds = pos_preds.squeeze(0).cpu().numpy()
        # 限制位置预测在1到pos_range之间
        pred_moves = [int(np.clip(p, 1, pos_range)) for p in pos_preds]

        best_moves = [[int(order[i]), pred_moves[i]] for i in range(3)]
        pred_score = calculate_score(A, B, order, pred_moves)

        return best_moves, pred_score
    except Exception as e:
        print(f"Error occurred with inputs A: {A} and B: {B} in DNN")
        print(f"Error message: {str(e)}")
        raise



def train():
    """
    加载训练数据并启动训练过程
    """
    try:
        with open(jsonfilename, "r") as f:
            train_data = json.load(f)
        print(f"成功加载训练数据，样本数: {len(train_data)}")
    except FileNotFoundError:
        print("未找到json文件，请确保文件存在")
        exit(1)

    train_model(train_data, epochs=10000, batch_size=1024, model_path="./trained/move_predictor.pth")


if __name__ == "__main__":
    train()
    # A= [1, 3, 7, 6, 2, 9]
    # B= [6, 8, 9]
    # [[1, 1], [2, 1], [0, 3]]
    # print(DNNpredict(A, B))
    # A=[4, 9, 11, 2, 13, 8]
    # B=[8, 8, 11]
    # best_moves, score = DNNpredict(A, B, model_path="./trained/move_predictor.pth")
    # print(f"\nA: {A}")
    # print(f"B: {B}")
    # print(f"预测最佳移动: {best_moves}")
    # print(f"预测得分: {score}")
    # 以下代码为额外测试，默认注释，若需要可取消注释
    # for _ in range(10):
    #
    #
    #     A, B = deal_cards_tool()
    #     best_moves, score = DNNpredict(A, B, model_path="move_predictor.pth")
    #     print(f"\nA: {A}")
    #     print(f"B: {B}")
    #     print(f"预测最佳移动: {best_moves}")
    #     print(f"预测得分: {score}")
    #     rec_score,rec_moves = recursive_strategy(A, B)
    #     print(f"递归算法得分: {rec_score}")
    #     print(f"递归算法移动: ",rec_moves)
