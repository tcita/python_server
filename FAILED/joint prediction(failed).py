import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 文件路径
openfilepaths = ["dataset_1.json", "dataset_2.json", "dataset_3.json"]
savefilepaths = ["best_move_transformer_part1.pth", "best_move_transformer_part2.pth",
                 "best_move_transformer_part3.pth"]

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据加载和预处理函数
def load_and_preprocess_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    A_values, B_values, best_moves = [], [], []
    for sample in data:
        A_values.append(sample["A"])
        B_values.append(sample["B"])
        best_moves.append(sample["best_moves"])

    max_A_length, max_B_length = 9, 3
    X_A = np.array([pad_array(a, max_A_length) for a in A_values])
    X_B = np.array([pad_array(b, max_B_length) for b in B_values])
    X = np.concatenate([X_A, X_B], axis=1)  # 形状: [samples, 12]

    # 重塑为序列，12 个特征分为 4 个时间步，每步 3 维
    X = X.reshape(-1, 4, 3)  # 形状: [samples, seq_len=4, feature_dim=3]

    y = np.array([move[0] * max_A_length + move[1] for move in best_moves])

    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 12)).reshape(-1, 4, 3)  # 归一化后恢复形状
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


def pad_array(arr, max_length, pad_value=0):
    return arr + [pad_value] * (max_length - len(arr))


# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)  # feature_dim=3
        self.positional_encoding = nn.Parameter(torch.zeros(1, 4, d_model))  # seq_len=4
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x 的形状: [batch, seq_len=4, feature_dim=3]
        x = self.embedding(x) + self.positional_encoding  # [batch, 4, d_model]
        x = self.transformer_encoder(x)  # [batch, 4, d_model]
        x = x.mean(dim=1)  # 池化得到 [batch, d_model]
        x = self.fc(x)  # [batch, num_classes]
        return x


# 联合训练函数
def train_jointly(models, X_trains, y_trains, criterion, optimizer, num_epochs=8000):
    for epoch in range(num_epochs):
        total_loss = 0
        for model, X_train, y_train in zip(models, X_trains, y_trains):
            model.train()
            X_train_device = torch.tensor(X_train, dtype=torch.float32).to(device)  # [batch, 4, 3]
            y_train_device = torch.tensor(y_train, dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(X_train_device)
            loss = criterion(outputs, y_train_device)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss:.4f}")

    # 测试每个模型的准确率
    for i, (model, X_test, y_test) in enumerate(zip(models, X_tests, y_tests)):
        model.eval()
        with torch.no_grad():
            X_test_device = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_device = torch.tensor(y_test, dtype=torch.long).to(device)
            outputs = model(X_test_device)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_device).sum().item() / y_test_device.size(0)
            print(f"Model {i + 1} Test accuracy: {accuracy:.4f}")


# 修改后的预测函数
def predict_best_move(A, B, loaded_model, scaler, used_b_indices=None):
    if used_b_indices is None:
        used_b_indices = []

    model = TransformerModel(feature_dim=3, num_classes=27).to(device)
    model.load_state_dict(torch.load(loaded_model, weights_only=True))
    model.eval()

    max_A_length, max_B_length = 9, 3
    A_padded = pad_array(A, max_A_length)
    B_padded = pad_array(B, max_B_length)
    X_input = np.concatenate([np.array([A_padded]), np.array([B_padded])], axis=1)  # [1, 12]
    X_input = X_input.reshape(1, 4, 3)  # [1, seq_len=4, feature_dim=3]
    X_input = scaler.transform(X_input.reshape(1, 12)).reshape(1, 4, 3)  # 归一化
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_input)  # [1, 27]
        # 屏蔽已使用的 B_index
        for b in used_b_indices:
            for a in range(max_A_length):  # 对于每个可能的 A_index
                class_idx = b * max_A_length + a
                outputs[0, class_idx] = float('-inf')  # 将对应的类别置为负无穷

        _, predicted = torch.max(outputs, 1)  # 选择未被屏蔽的最大值

    B_index, A_index = divmod(predicted.item(), max_A_length)
    return [B_index, A_index]


# 主函数
if __name__ == "__main__":
    # 加载所有数据集
    X_trains, X_tests, y_trains, y_tests, scalers = [], [], [], [], []
    for filepath in openfilepaths:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(filepath)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        scalers.append(scaler)

    # 创建多个模型
    models = [TransformerModel(feature_dim=3, num_classes=27).to(device) for _ in range(3)]

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([param for model in models for param in model.parameters()], lr=0.001, weight_decay=1e-4)

    # 联合训练
    train_jointly(models, X_trains, y_trains, criterion, optimizer)

    # 保存模型
    for model, savefilepath in zip(models, savefilepaths):
        torch.save(model.state_dict(), savefilepath)
        print(f"Model saved to {savefilepath}")

    # 示例预测
    from AI_algorithm.test.testtool import deal_cards_tool
    from AI_algorithm.brute_force import recursive_strategy
    from AI_algorithm.GA import simulate_insertion

    A, B = deal_cards_tool()
    print("A:", A)
    print("B:", B)
    _, move = recursive_strategy(A, B)
    print("True move:", move if move else [0, 1])

    print("-" * 20)
    print("Predicted moves:")
    loaded_models = savefilepaths
    newA = A.copy()
    used_b_indices = []  # 记录已使用的 B_index
    for i, loaded_model in enumerate(loaded_models):
        b, A_pos = predict_best_move(newA, B, loaded_model, scalers[i], used_b_indices)
        used_b_indices.append(b)  # 添加当前 B_index 到已使用列表
        print(f"Model {i + 1}: [{b}, {A_pos}]")
        score, _, _, _, newA = simulate_insertion(newA, b, A_pos)
        print(f"New A after Model {i + 1}: {newA}")
        print(f"Score {i + 1}: {score}")