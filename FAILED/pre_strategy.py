import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from AI_algorithm.GA import simulate_insertion
from AI_algorithm.brute_force import recursive_strategy
from AI_algorithm.test.testtool import deal_cards_tool
#
openfilepath = "dataset_1.json"
savefilepath = "best_move_transformer_part1.pth"

# openfilepath = "dataset_2.json"
# savefilepath = "best_move_transformer_part2.pth"
#
# openfilepath = "dataset_3.json"
# savefilepath = "best_move_transformer_part3.pth"
import torch
print(torch.__version__)  # 查看 PyTorch 版本
print(torch.version.cuda)  # 检查 CUDA 版本（如果是 None，说明没有安装 CUDA 版）
print(torch.cuda.is_available())  # 如果是 False，说明 PyTorch 没有检测到 GPU



# 选择设备，如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 从 data.json 文件加载数据
with open(openfilepath, "r") as f:
    data = json.load(f)

# 提取 A, B 和 best_moves
A_values, B_values, best_moves = [], [], []
for sample in data:
    A_values.append(sample["A"])
    B_values.append(sample["B"])
    best_moves.append(sample["best_moves"])

# A 长度为 6，B 长度为 3
max_A_length, max_B_length = 9, 3


def pad_array(arr, max_length, pad_value=0):
    return arr + [pad_value] * (max_length - len(arr))


# 填充 A 和 B
X_A = np.array([pad_array(a, max_A_length) for a in A_values])
X_B = np.array([pad_array(b, max_B_length) for b in B_values])
y = np.array(best_moves)

# 合并 A 和 B 数据作为输入
X = np.concatenate([X_A, X_B], axis=1)

# 归一化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 编码输出
y = np.array([move[0] * max_A_length + move[1] for move in y])

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为 PyTorch 张量
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 池化得到固定维度的向量
        x = self.fc(x)
        return x


def train():
    # 创建模型并移动到设备
    model = TransformerModel(input_dim=max_A_length + max_B_length, num_classes=27).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train()

        # 将数据移动到设备
        X_train_device = X_train.to(device)
        y_train_device = y_train.to(device)

        optimizer.zero_grad()
        outputs = model(X_train_device.unsqueeze(1))
        loss = criterion(outputs, y_train_device)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        y_test_device = y_test.to(device)
        outputs = model(X_test_device.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_device).sum().item() / y_test_device.size(0)
        print(f"Test accuracy: {accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), savefilepath)
    print("Model saved to best_move_transformer.pth")

# loaded_model=""
def predict_best_move(A, B,loaded_model):
    # 加载训练好的模型
    model = TransformerModel(input_dim=max_A_length + max_B_length, num_classes=27).to(device)
    model.load_state_dict(torch.load(loaded_model, weights_only=True))
    model.eval()

    A_padded = pad_array(A, max_A_length)
    B_padded = pad_array(B, max_B_length)
    X_input = np.concatenate([np.array([A_padded]), np.array([B_padded])], axis=1)
    X_input = scaler.transform(X_input)  # 归一化
    X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_input)
        _, predicted = torch.max(outputs, 1)

    B_index, A_index = divmod(predicted.item(), max_A_length)
    return [B_index, A_index]


if __name__ == '__main__':
    # train()
    loaded_model1="best_move_transformer_part1.pth"
    loaded_model2 = "best_move_transformer_part2.pth"
    loaded_model3 = "best_move_transformer_part3.pth"
    A,B=deal_cards_tool()
    print("A")
    print(A)
    print("B")
    print(B)

    _,move=recursive_strategy(A,B)
    print("ture move:")
    if move == []:
        print("[0, 1]")
    else:
        print(move)
    print("-" * 20)
    print("-" * 20)

    print("predicted:")
    b,A_pos=predict_best_move(A, B, loaded_model1)
    print(b,A_pos)
    score1, _, _, _, newA = simulate_insertion(A, b, A_pos)
    print("newA")
    print(newA)
    b, A_pos = predict_best_move(newA, B, loaded_model2)

    score2, _, _, _, newA = simulate_insertion(A, b, A_pos)

    print(b,A_pos)

    b, A_pos = predict_best_move(newA, B, loaded_model3)
    print(b, A_pos)


    print("score1:",score1)
    print("newA:",newA)
