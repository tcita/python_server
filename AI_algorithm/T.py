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

class TMovePredictor(nn.Module):
    def __init__(self, input_size_A: int, input_size_B: int):
        super(TMovePredictor, self).__init__()

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

        # 位置编码路径
        self.fc_pos = nn.Linear(7, 64)  # 位置编码的维度（7：即最大插入位置）

        # 融合A、B和位置编码的特征
        # num_b是3，那么x_pos的维度将会是batch_size * 3 * 64 = batch_size * 192。在这种情况下，拼接后的
        # x的维度将会是batch_size * (64 + 64 + 192) = batch_size * 320

        self.fc3 = nn.Linear(320, 32)

        self.bn3 = nn.BatchNorm1d(32)

        # 输出层
        self.order_head = nn.Linear(32, 9)
        self.pos_head = nn.Linear(32, 3)

    def forward(self, A, B, position_encodings):
        # 处理A的路径
        x_A = torch.relu(self.bn1_A(self.fc1_A(A)))
        x_A = torch.relu(self.bn2_A(self.fc2_A(x_A)))

        # 处理B的路径
        x_B = torch.relu(self.bn1_B(self.fc1_B(B)))
        x_B = torch.relu(self.bn2_B(self.fc2_B(x_B)))

        # 处理位置编码的路径
        x_pos = torch.relu(self.fc_pos(position_encodings))

        # print(f"x_A shape: {x_A.shape}")
        # print(f"x_B shape: {x_B.shape}")
        # print(f"x_pos shape: {x_pos.shape}")

        # 如果 x_pos 是三维的，展平为二维
        if x_pos.dim() == 3:
            x_pos = x_pos.view(x_pos.size(0), -1)  # 展平为 (batch_size, 192)
        elif x_pos.dim() == 2:
            pass  # 已经是二维，不需要做额外处理
        else:
            raise ValueError(f"Unexpected dimension for x_pos: {x_pos.dim()}")

        # 合并A、B和位置编码的特征
        x = torch.cat([x_A, x_B, x_pos], dim=-1)  # 拼接A、B和位置编码的特征

        # 确保拼接后的维度与fc3匹配
        # print(f"拼接后的特征维度: {x.shape}")

        # 经过全连接层
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
    A = sample["A"]
    B = sample["B"]
    best_moves = sample["best_moves"]

    # 固定A和B的长度
    assert len(A) == 6, f"A的长度应该是6，但实际是{len(A)}"
    assert len(B) == 3, f"B的长度应该是3，但实际是{len(B)}"

    # 为每个B元素分配一个动态位置编码
    position_encodings = []
    for i in range(len(B)):
        # 对于每个B[i]，它可以插入到位置 0 到 len(A) 之间
        position_encoding = [0] * (len(A) + 1)  # 位置编码范围是 0 到 len(A)
        position_encoding[i] = 1  # 在插入位置处设置为1
        position_encodings.append(position_encoding)

    # 将A和B拼接成输入向量
    input_vector_A = np.array(A, dtype=np.float32)
    input_vector_B = np.array(B, dtype=np.float32)

    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)

    return input_vector_A, input_vector_B, order_target, pos_target, position_encodings

def train_model(train_data, epochs=2000, batch_size=512, model_path="./trained/T.pth"):
    input_size_A = 6  # A的维度
    input_size_B = 3  # B的维度
    model = TMovePredictor(input_size_A, input_size_B).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            inputs_A, inputs_B, order_targets, pos_targets, position_encodings = [], [], [], [], []

            for sample in batch:
                input_A, input_B, order_tgt, pos_tgt, pos_enc = prepare_data(sample)
                if input_A is None or input_B is None:  # 跳过噪音样本
                    continue
                inputs_A.append(input_A)
                inputs_B.append(input_B)
                order_targets.append(order_tgt)
                pos_targets.append(pos_tgt)
                position_encodings.append(pos_enc)

            if not inputs_A:
                continue

            inputs_A = torch.FloatTensor(np.array(inputs_A)).to(device)
            inputs_B = torch.FloatTensor(np.array(inputs_B)).to(device)
            order_targets = torch.LongTensor(np.array(order_targets)).to(device)
            pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device)
            position_encodings = torch.FloatTensor(np.array(position_encodings)).to(device)

            # 将位置编码传递给模型
            order_logits, pos_preds = model(inputs_A, inputs_B, position_encodings)

            order_loss = order_criterion(order_logits.view(-1, 3), order_targets.view(-1))
            pos_loss = pos_criterion(pos_preds, pos_targets)
            loss = order_loss + pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # 打印当前batch的进度
            if batch_count % 10 == 0:
                conditional_print(f"Epoch {epoch}, Batch {batch_count}, Loss: {loss.item():.4f}")

        # 每100个epoch打印一次平均损失
        if epoch % 100 == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']}")


    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")
    return model

def TDNNpredict(A, B, model):
    """
    根据输入的 A 和 B，利用训练好的神经网络模型预测最佳移动和得分
    :param A: 列表，表示当前状态
    :param B: 列表，表示可选操作
    :param model: 已经加载好的 TMovePredictor 模型实例
    :return: 最佳策略和得分
    """
    try:
        if not set(A) & set(B) and len(B) == len(set(B)):
            return [[0, 1], [1, 1], [2, 1]], 0

        if not isinstance(model, TMovePredictor):
            raise ValueError(f"Expected a TMovePredictor instance, but got {type(model)}")

        model.eval()

        # 准备输入数据
        input_vector_A = np.array(A, dtype=np.float32)
        input_vector_B = np.array(B, dtype=np.float32)

        # 生成位置编码
        position_encodings = []
        for i in range(len(B)):
            position_encoding = [0] * (len(A) + 1)  # 位置编码范围是 0 到 len(A)
            position_encoding[i] = 1  # 在插入位置处设置为1
            position_encodings.append(position_encoding)

        # Convert to Tensor and add batch dimension
        input_tensor_A = torch.FloatTensor(input_vector_A).unsqueeze(0).to(device)
        input_tensor_B = torch.FloatTensor(input_vector_B).unsqueeze(0).to(device)
        position_encodings_tensor = torch.FloatTensor(np.array(position_encodings)).unsqueeze(0).to(device) # Add unsqueeze(0) here

        with torch.no_grad():
            # 获取模型输出
            order_logits, pos_preds = model(input_tensor_A, input_tensor_B, position_encodings_tensor)

        # 从logits中获取排序顺序
        order = sample_order(order_logits).squeeze(0).cpu().numpy()

        # 处理位置预测（确保预测的位置范围不超出有效范围）
        pos_range = len(A)
        pos_preds = pos_preds.squeeze(0).cpu().numpy()
        pred_moves = [int(np.clip(p, 0, pos_range)) for p in pos_preds]

        # 计算最佳移动
        best_moves = [[int(order[i]), pred_moves[i]] for i in range(3)]

        # 计算预测的得分
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

    train_model(train_data, epochs=3000, batch_size=1024, model_path="./trained/T.pth")


if __name__ == "__main__":
    train()