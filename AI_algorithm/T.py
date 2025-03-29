import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from AI_algorithm.tool.tool import calculate_score_by_strategy

# 两个向量组成一个会丢失特征吗?
jsonfilename = "json/data_raw.json"
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
    strategy = [order[0], moves[0]], [order[1], moves[1]], [order[2], moves[2]]
    total_score = calculate_score_by_strategy(A, B, strategy)
    return total_score


class MovePredictor(nn.Module):
    def __init__(self, input_size_A: int, input_size_B: int):
        super(MovePredictor, self).__init__()

        # A的网络路径
        self.fc1_A = nn.Linear(input_size_A, 128)
        self.bn1_A = nn.BatchNorm1d(128)
        self.fc2_A = nn.Linear(128, 64)
        self.bn2_A = nn.BatchNorm1d(64)

        # B的网络路径
        self.fc1_B = nn.Linear(input_size_B, 128)
        self.bn1_B = nn.BatchNorm1d(128)
        self.fc2_B = nn.Linear(128, 64)
        self.bn2_B = nn.BatchNorm1d(64)

        # 融合A和B的特征
        self.fc3 = nn.Linear(64 + 64, 32)  # A和B的特征被拼接在一起
        self.bn3 = nn.BatchNorm1d(32)

        # 输出层
        self.order_head = nn.Linear(32, 9)
        self.pos_head = nn.Linear(32, 3)

    def forward(self, A, B):
        # 处理A的路径
        x_A = torch.relu(self.bn1_A(self.fc1_A(A)))
        x_A = torch.relu(self.bn2_A(self.fc2_A(x_A)))

        # 处理B的路径
        x_B = torch.relu(self.bn1_B(self.fc1_B(B)))
        x_B = torch.relu(self.bn2_B(self.fc2_B(x_B)))

        # 合并A和B的特征
        x = torch.cat([x_A, x_B], dim=-1)  # 拼接两个向量

        x = torch.relu(self.bn3(self.fc3(x)))

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
    best_moves = sample["best_moves"]

    # 填充A和B到9个元素
    # 如果A或B的长度小于9，用0填充
    if len(A) < 9:
        A = A + [0] * (9 - len(A))
    if len(B) < 9:
        B = B + [0] * (9 - len(B))

    # 如果A或B的长度大于9，裁剪为前9个元素
    A = A[:9]
    B = B[:9]

    # 将A和B拼接成输入向量
    input_vector_A = np.array(A, dtype=np.float32)
    input_vector_B = np.array(B, dtype=np.float32)

    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)

    return input_vector_A, input_vector_B, order_target, pos_target



def train_model(train_data, epochs=2000, batch_size=512, model_path="./trained/T.pth"):
    input_size_A = 9  # A的维度
    input_size_B = 9  # B的维度
    model = MovePredictor(input_size_A, input_size_B).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            inputs_A, inputs_B, order_targets, pos_targets = [], [], [], []

            for sample in batch:
                input_A, input_B, order_tgt, pos_tgt = prepare_data(sample)
                if input_A is None or input_B is None:  # 跳过噪音样本
                    continue
                inputs_A.append(input_A)
                inputs_B.append(input_B)
                order_targets.append(order_tgt)
                pos_targets.append(pos_tgt)

            if not inputs_A:
                continue

            inputs_A = torch.FloatTensor(np.array(inputs_A)).to(device)
            inputs_B = torch.FloatTensor(np.array(inputs_B)).to(device)
            order_targets = torch.LongTensor(np.array(order_targets)).to(device)
            pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device)

            order_logits, pos_preds = model(inputs_A, inputs_B)

            order_loss = order_criterion(order_logits.view(-1, 3), order_targets.view(-1))
            pos_loss = pos_criterion(pos_preds, pos_targets)
            loss = order_loss + pos_loss

            optimizer.zero_grad()
            loss.backward()
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
        if not set(A) & set(B) and len(B) == len(set(B)):
            return [[0, 1], [1, 1], [2, 1]], 0

        if not isinstance(model, MovePredictor):
            raise ValueError("Expected a MovePredictor instance, but got {}".format(type(model)))

        model.eval()
        input_vector_A = np.array(A, dtype=np.float32)
        input_vector_B = np.array(B, dtype=np.float32)

        input_tensor_A = torch.FloatTensor(input_vector_A).unsqueeze(0).to(device)
        input_tensor_B = torch.FloatTensor(input_vector_B).unsqueeze(0).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor_A, input_tensor_B)

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

    train_model(train_data, epochs=2500, batch_size=1024, model_path="./trained/T.pth")


if __name__ == "__main__":
    train()