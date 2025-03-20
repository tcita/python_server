import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from AI_algorithm.transformer_score import transformer_scoreonly


# 改进后的 Transformer 模型
class BestMovesTransformer(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(BestMovesTransformer, self).__init__()

        self.embedding = nn.Linear(10, d_model)  # A(6) + B(3) + max_score(1) = 10 输入特征
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 修改这里，启用 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 输出最佳的3对 (B索引, 插入位置)
        )

    def forward(self, x):
        x = self.embedding(x)  # 将输入映射到 d_model
        x = self.pos_encoder(x)  # 加入位置编码
        x = self.encoder(x)  # Transformer 编码器
        x = self.decoder(x)  # 输出最佳动作
        return x.view(-1, 3, 2)  # 重塑为 (batch_size, 3, 2)


# 自定义数据集
class TrainingDataset(Dataset):
    def __init__(self, file_path, inference=False):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.inference = inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        A = torch.tensor(sample['A'], dtype=torch.float32)
        B = torch.tensor(sample['B'], dtype=torch.float32)

        if self.inference:
            # 预测的值
            max_score = round(transformer_scoreonly(sample['A'], sample['B']))  # 取整
        else:
            max_score = round(sample['max_score'])  # 取整

        max_score = torch.tensor([max_score], dtype=torch.float32)  # 转换成张量

        best_moves = sample.get('best_moves', [[0, 0], [0, 0], [0, 0]])
        best_moves = torch.tensor(best_moves, dtype=torch.long)

        x = torch.cat((A, B, max_score))
        return x, best_moves


# Collate 函数，处理批次数据
def collate_fn(batch):
    inputs, targets = zip(*batch)

    # 填充目标值，使其匹配 [3, 2] 的形状
    targets = [t if t.size(0) > 0 else torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.long) for t in targets]

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets


# 改进后的损失函数
def custom_loss_fn(y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true.float())
    punish=0
    penalties = torch.tensor(0.0, device=y_pred.device)

    for pred in y_pred:
        first_elements = pred[:, 0]
        unique_first_elements = torch.unique(first_elements)

        # 适当降低惩罚值
        if unique_first_elements.size(0) < 3 or torch.any((first_elements < 0) | (first_elements > 2)):
            penalties += punish  # 降低惩罚值

    second_elements = y_pred[:, :, 1]
    penalty_for_invalid_second_values = torch.where(
        (second_elements < 1) | (second_elements > 8),
        torch.tensor(punish, device=y_pred.device),  # 适当降低惩罚值
        torch.tensor(0.0, device=y_pred.device)
    ).sum()

    first_element_is_int = torch.abs(y_pred[:, :, 0] - torch.round(y_pred[:, :, 0])) < 1e-3
    second_element_is_int = torch.abs(y_pred[:, :, 1] - torch.round(y_pred[:, :, 1])) < 1e-3

    penalty_for_non_integer = torch.where(
        ~(first_element_is_int & second_element_is_int),
        torch.tensor(punish, device=y_pred.device),  # 降低惩罚
        torch.tensor(0.0, device=y_pred.device)
    ).sum()

    total_penalty = penalties + penalty_for_invalid_second_values + penalty_for_non_integer

    return mse_loss + total_penalty


# 训练函数，增加了学习率调度器
# 训练函数，增加了学习率调度器和梯度裁剪
def train_model(model, dataloader, epochs=200, lr=0.001, save_path="best_moves_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # 学习率调度器

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred = model(x)
            loss = custom_loss_fn(y_pred, y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        # 调整学习率
        scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 加载训练好的模型
def load_model(model_path="best_moves_model.pth"):
    model = BestMovesTransformer()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型
    model.eval()  # 设置为评估模式
    return model


def evaluate_model(model, A, B):
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    max_score = torch.tensor([transformer_scoreonly(A.tolist(), B.tolist())], dtype=torch.float32)

    x = torch.cat((A, B, max_score)).unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        y_pred = model(x)

    return y_pred.squeeze(0).tolist()  # 转换为二维数组并返回

if __name__ == '__main__':
    # 加载数据集并训练模型
    dataset = TrainingDataset('data.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = BestMovesTransformer()
    train_model(model, dataloader)
