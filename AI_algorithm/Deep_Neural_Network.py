import time

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from AI_algorithm.tool.tool import calculate_score_by_strategy

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
    """
    从 logits 采样顺序，确保输出为 0,1,2 的一个排列
    """
    probs = torch.softmax(logits / temperature, dim=-1)  # 计算概率
    order = torch.argsort(-probs, dim=-1)  # 概率最高的优先
    order = order[:, :, 0].squeeze()  # 取每个样本的最大概率索引

    # 确保 order 是二维张量，即使 batch_size=1
    if order.dim() == 1:
        order = order.unsqueeze(0)

    # 确保 order 是 0,1,2 的一个排列
    batch_size = order.shape[0]
    valid_orders = torch.stack([torch.randperm(3) for _ in range(batch_size)]).to(order.device)

    # 使用张量操作代替列表推导式
    mask = torch.tensor([len(set(row.tolist())) != 3 for row in order], device=order.device)
    order[mask] = valid_orders[mask]  # 如果 order 不是 0,1,2 的排列，则随机重新赋值

    return order

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

    optimizer = optim.Adam(model.parameters(), lr=0.01)
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

            pos_targets = [pos_tgt  for pos_tgt in pos_targets]
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


def DNNpredict(A, B, model):
    """
    根据输入的 A 和 B，利用训练好的神经网络模型预测最佳移动和得分
    :param A: 列表，表示当前状态
    :param B: 列表，表示可选操作
    :param model: 已经加载好的 MovePredictor 模型实例
    :return: 最佳策略和得分
    """
    try:
        # 如果 A 和 B 没有交集，且 B 中没有重复元素，直接返回固定策略
        if not set(A) & set(B) and len(B) == len(set(B)):
            return [[0, 0], [1, 0], [2, 0]], 0

        # 确保传入的 model 是 MovePredictor 实例
        if not isinstance(model, MovePredictor):
            raise ValueError("Expected a MovePredictor instance, but got {}".format(type(model)))

        model.eval()  # 设置为评估模式
        input_vector = np.array(A + B, dtype=np.float32)
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        # 确保 order 是 0, 1, 2 的一个排列
        order = sample_order(order_logits).squeeze(0).cpu().numpy()

        pos_range = len(A)
        pos_preds = pos_preds.squeeze(0).cpu().numpy()
        pred_moves = [int(np.clip(p, 0, pos_range)) for p in pos_preds]

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

    train_model(train_data, epochs=2500, batch_size=1024, model_path="./trained/move_predictor.pth")


if __name__ == "__main__":
    train()