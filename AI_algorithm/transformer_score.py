from pathlib import Path

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# 由于训练集的len(A)与len(B)的长度是固定的  所以当预测其他长度的A，B 会不再可靠

# -----------------------------
# 数据集定义与预处理
# -----------------------------
class TrainingDataset(Dataset):
    def __init__(self, file_path):
        # 加载 JSON 数据，假设文件中是一个列表，每个元素为一条记录
        with open(file_path, 'r') as f:
            self.samples = json.load(f)

        # 定义特殊 token 的索引
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.sep_token = 3
        # 数字 token 的偏移量，保证不会和特殊 token 冲突
        self.num_offset = 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        A = sample['A']
        B = sample['B']
        max_score = sample['max_score']
        best_moves = sample['best_moves']  # list of pairs

        # 将 A 和 B 中的数字映射到 token（加上偏移量）
        A_tokens = [x + self.num_offset for x in A]
        B_tokens = [x + self.num_offset for x in B]
        # 输入序列：A_tokens + [SEP] + B_tokens
        src = A_tokens + [self.sep_token] + B_tokens

        # 将 best_moves 平铺为一维序列，例如 [[1,1],[2,5],[0,5]] -> [1,1,2,5,0,5]
        moves_flat = [num + self.num_offset for pair in best_moves for num in pair]
        # 输出序列： [SOS] + moves_flat + [EOS]
        trg = [self.sos_token] + moves_flat + [self.eos_token]

        return {
            'src': torch.tensor(src, dtype=torch.long),
            'trg': torch.tensor(trg, dtype=torch.long),
            'score': torch.tensor([max_score], dtype=torch.float)
        }


def collate_fn(batch):
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]
    score_batch = torch.stack([item['score'] for item in batch])

    # 对序列做 padding，padding_value 使用 PAD token（0）
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return {'src': src_batch, 'trg': trg_batch, 'score': score_batch}


# -----------------------------
# 模型定义
# -----------------------------
# Encoder 定义
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1):
        """
        input_size: 输入词典大小
        embed_size: 嵌入维度
        hidden_size: LSTM隐藏层维度
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, embed_size)
        outputs, (hidden, cell) = self.lstm(embed)  # outputs: (batch, seq_len, hidden_size)
        return outputs, hidden, cell


# Decoder 定义
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1):
        """
        output_size: 输出词典大小（包括所有可能的数字和特殊 token）
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        # input: (batch) 单个 token 索引，扩展为 (batch, 1)
        input = input.unsqueeze(1)
        embed = self.embedding(input)  # (batch, 1, embed_size)
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))  # output: (batch, 1, hidden_size)
        prediction = self.out(output.squeeze(1))  # (batch, output_size)
        return prediction, hidden, cell


# 整体 Seq2Seq 模型，既输出动作序列又预测得分
# Seq2Seq 模型定义中的 score_head 部分
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, score_hidden_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # 通过编码器最后一层的隐藏状态预测 max_score
        self.score_head = nn.Sequential(
            nn.Linear(score_hidden_size, score_hidden_size // 2),
            nn.ReLU(),  # 使用 ReLU 激活函数，确保得分非负
            nn.Linear(score_hidden_size // 2, 1),
            nn.ReLU()  # 再次应用 ReLU 确保最终的得分是非负的
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码器处理输入
        encoder_outputs, hidden, cell = self.encoder(src)

        # 通过编码器最后的隐藏状态预测得分，确保得分非负
        score_pred = self.score_head(hidden[-1])

        # 初始化解码器输入：使用 trg 中的第一个 token (<SOS>)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            # 是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs, score_pred


# -----------------------------
# 超参数与模型实例化
# -----------------------------
# 根据数据中数字的范围和特殊 token 定义词典大小
# 例如：A 和 B 中的数字较小，则输入词典大小设置为 100
INPUT_SIZE = 100
# 输出词典大小设置为 50（依据 best_moves 中数字范围和特殊 token）
OUTPUT_SIZE = 50
EMBED_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(__file__).parent / 'json' / 'data_raw.json'
dataset = TrainingDataset(str(file_path))

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=True)

# -----------------------------
# 初始化模型、优化器及损失函数
# -----------------------------
encoder_model = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
decoder_model = Decoder(OUTPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
model = Seq2Seq(encoder_model, decoder_model, HIDDEN_SIZE, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)
# 对动作序列采用交叉熵损失（忽略 <SOS>，从 t=1 开始）
criterion = nn.CrossEntropyLoss()
# 得分预测采用均方误差损失
score_criterion = nn.MSELoss()


# -----------------------------
# 训练函数
# -----------------------------
# 训练函数中的得分计算部分
def train():
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            # 使用 non_blocking=True 加速数据传输
            src = batch['src'].to(DEVICE, non_blocking=True)
            true_score = batch['score'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            # 使用自动混合精度（仅在 CUDA 可用时）
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):

                _, score_pred = model(src, src)  # 这里 trg 其实不需要了
                loss = score_criterion(score_pred, true_score)

            if scaler is not None:
                scaler.scale(loss).backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "trained/seq2seq_model.pth")
    print("模型已保存到 trained/seq2seq_model.pth")
# -----------------------------
# 推断函数：用 A, B 测试模型
# -----------------------------
# 推理函数中的得分限制部分
def predict(model, A, B, num_offset=4, sep_token=3):
    model.eval()
    with torch.no_grad():
        A_tokens = [x + num_offset for x in A]
        B_tokens = [x + num_offset for x in B]
        src = A_tokens + [sep_token] + B_tokens
        src_tensor = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # 编码阶段
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        score_pred = model.score_head(hidden[-1])
        score_pred = torch.clamp(score_pred, min=0).item()  # 确保非负

    return score_pred


# -----------------------------
# 加载模型功能
# -----------------------------
def load_model(model_path):
    """
    加载已保存的模型，并返回加载好的模型
    """
    encoder_model = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    decoder_model = Decoder(OUTPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    loaded_model = Seq2Seq(encoder_model, decoder_model, HIDDEN_SIZE, DEVICE).to(DEVICE)
    loaded_model.load_state_dict(torch.load(model_path, map_location=DEVICE,weights_only=True))
    loaded_model.eval()
    return loaded_model


# -----------------------------
# 定义 transformer_scoreonly 接口
# -----------------------------
def transformer_scoreonly(A_test, B_test,model_path="../trained/seq2seq_model.pth"):
    """
    输入 A_test, B_test，返回模型预测的得分
    """

    loaded_model = load_model(model_path)
    predicted_score = predict(loaded_model, A_test, B_test)
    return predicted_score


# -----------------------------
# 主函数：训练或加载模型，并调用 modsummery 测试
# -----------------------------
if __name__ == '__main__':
    model_path = "trained/seq2seq_model.pth"
    # 如果模型文件不存在则训练，否则加载模型
    if not os.path.exists(model_path):
        print("模型文件不存在，开始训练...")
        train()
    else:
        print("检测到模型文件，直接加载模型。")

    loaded_model = load_model(model_path)
    print("模型已加载。")

    # 调用 modsummery 对 transformer_scoreonly 进行测试（例如测试2000次）
    # modsummery(transformer_scoreonly, 2000)
# 假设已经加载了模型，存储在变量 loaded_model 中
# A_test = [1, 12, 1]   # 示例 A 列表
# B_test = [7, 19]   # 示例 B 列表
#
# # 调用 predict 函数，返回最佳移动序列和预测得分
# _, predicted_score = predict(loaded_model, A_test, B_test)
#
#
# true_score=recursive(A_test,B_test)
#
# genome = load_best_genome()
#
# GA_partial = partial(GA, genome)
# print("真实得分:",true_score)
# print("GA预测:", GA_partial(A_test,B_test))
# print("T预测:", predicted_score)改变代码
