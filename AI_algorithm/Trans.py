import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from AI_algorithm.tool.tool import calculate_score_by_strategy


# from AI_algorithm.tool.tool import calculate_score_by_strategy
# 为了代码能独立运行，这里提供一个模拟的函数



jsonfilename="json/data_raw.json" # 请确保你的 JSON 文件路径正确
GPUDEBUG_MODE = True

def conditional_print(*args, **kwargs):
    if GPUDEBUG_MODE:
        print(*args, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用，使用CPU")


def calculate_score(A, B, order, moves):
    strategy = [[int(order[i]), int(moves[i])] for i in range(len(order))]
    total_score = calculate_score_by_strategy(A, B, strategy)
    return total_score

# --- Transformer 相关组件 ---

class PositionalEncoding(nn.Module):
    # 确保 max_len >= num_a + num_b (6 + 3 = 9)
    def __init__(self, d_model, dropout=0.1, max_len=20): # 增加 max_len 以确保安全
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerMovePredictor(nn.Module):
    # 修改默认参数以反映新的固定长度
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3): # <--- 修改 num_a 默认值
        super(TransformerMovePredictor, self).__init__()
        # 验证传入的 num_a 和 num_b 是否符合预期（可选但推荐）
        if num_a != 6 or num_b != 3:
             print(f"警告: TransformerMovePredictor 初始化期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        self.seq_len = num_a + num_b # 现在是 6 + 3 = 9
        self.d_model = d_model

        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model) # 0 for A, 1 for B
        # 确保 PositionalEncoding 的 max_len 足够大
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len + 5)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 输出头仍然基于 num_b = 3
        self.order_head = nn.Linear(num_b * d_model, num_b * num_b)
        self.pos_head = nn.Linear(num_b * d_model, num_b)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.input_embed.bias.data.zero_()
        self.type_embed.weight.data.uniform_(-initrange, initrange)
        self.order_head.weight.data.uniform_(-initrange, initrange)
        self.order_head.bias.data.zero_()
        self.pos_head.weight.data.uniform_(-initrange, initrange)
        self.pos_head.bias.data.zero_()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if src.dim() == 2:
            src = src.unsqueeze(-1) # (batch, seq_len, 1)

        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)

        # type_ids: num_a 个 0, num_b 个 1
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1) # (batch_size, seq_len)
        type_embedded = self.type_embed(type_ids)

        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded)

        memory = self.transformer_encoder(embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # memory shape: (batch_size, seq_len=9, d_model)

        # 提取 B 对应的输出特征 (最后 num_b=3 个)
        # 使用 self.num_a (现在是6) 作为索引起点
        b_features = memory[:, self.num_a:] # (batch_size, num_b=3, d_model)

        b_features_flat = b_features.reshape(b_features.size(0), -1) # (batch_size, num_b * d_model)

        order_logits_flat = self.order_head(b_features_flat)
        order_logits = order_logits_flat.view(-1, self.num_b, self.num_b) # (batch_size, 3, 3)

        pos_preds = self.pos_head(b_features_flat) # (batch_size, 3)

        return order_logits, pos_preds

# --- 数据准备和训练逻辑 (调整以适应 Transformer) ---

def sample_order(logits, temperature=1.0):
    # (保持不变, 因为 B 的数量仍然是 3)
    probs = torch.softmax(logits / temperature, dim=-1)
    order_indices = torch.argsort(logits, dim=1, descending=True)
    order = order_indices[:, 0, :]

    if order.dim() == 1:
        order = order.unsqueeze(0)

    batch_size = order.shape[0]
    num_b = order.shape[1] # Should be 3
    valid_permutations = torch.stack([torch.randperm(num_b) for _ in range(batch_size)]).to(order.device)

    is_invalid = torch.zeros(batch_size, dtype=torch.bool, device=order.device)
    for b in range(batch_size):
        if len(torch.unique(order[b])) != num_b:
            is_invalid[b] = True

    order[is_invalid] = valid_permutations[is_invalid]
    return order


# 修改默认参数以反映新的固定长度
def prepare_data_transformer(sample: dict, num_a=6, num_b=3): # <--- 修改 num_a 默认值
    """
    数据预处理：构建输入序列和目标输出。
    """
    A = sample["A"]
    B = sample["B"]
    best_moves = sample["best_moves"]

    # 检查长度是否精确匹配 6 和 3
    if len(A) != num_a or len(B) != num_b:
         # 现在我们期望 A 长度为 6, B 长度为 3
         print(f"Warning: Skipping sample with A={A}, B={B} due to unexpected length (expected A:{num_a}, B:{num_b}).")
         return None, None, None

    input_sequence = np.array(A + B, dtype=np.float32) # (seq_len=9,)

    order_target = np.array([move[0] for move in best_moves], dtype=np.int64) # (num_b=3,)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32) # (num_b=3,)

    if len(set(order_target)) != num_b:
        print(f"Warning: Skipping sample with invalid order target in best_moves: {order_target}")
        return None, None, None

    return input_sequence, order_target, pos_target


# 修改 train_model 以使用新的固定长度
def train_model(train_data, epochs=1000, batch_size=64, model_path="./trained/transformer_move_predictor_6x3.pth", # <--- 修改模型保存路径名
                num_a=6, num_b=3): # <--- 明确设置 num_a=6
    """
    训练 Transformer 模型 (针对 A=6, B=3)
    """
    # 模型参数 (可以根据需要调整)
    d_model = 128
    nhead = 4 # d_model=128 可以被 nhead=4 整除
    num_encoder_layers = 3
    dim_feedforward = 256
    dropout = 0.1

    # 初始化模型时传入正确的 num_a, num_b
    model = TransformerMovePredictor(input_dim=1, d_model=d_model, nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     dim_feedforward=dim_feedforward, dropout=dropout,
                                     num_a=num_a, num_b=num_b).to(device) # <--- 传递 num_a=6

    conditional_print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    conditional_print("模型是否在GPU上:", next(model.parameters()).is_cuda)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            inputs, order_targets, pos_targets = [], [], []

            for sample in batch:
                # 调用 prepare_data 时传递正确的 num_a, num_b
                input_seq, order_tgt, pos_tgt = prepare_data_transformer(sample, num_a=num_a, num_b=num_b) # <--- 传递 num_a=6
                if input_seq is None:
                    continue
                inputs.append(input_seq)
                order_targets.append(order_tgt)
                pos_targets.append(pos_tgt)

            if not inputs:
                continue

            inputs = torch.FloatTensor(np.array(inputs)).to(device) # (batch, seq_len=9)
            order_targets = torch.LongTensor(np.array(order_targets)).to(device) # (batch, num_b=3)
            pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device) # (batch, num_b=3)

            optimizer.zero_grad()
            order_logits, pos_preds = model(inputs)
            # order_logits: (batch, 3, 3), pos_preds: (batch, 3)

            order_loss = order_criterion(order_logits, order_targets)
            pos_loss = pos_criterion(pos_preds, pos_targets)
            loss = order_loss + pos_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Transformer 模型 (6x3) 已保存至 {model_path}")
    return model


# 修改 DNNpredict 以使用新的固定长度
def Transformer_predict(A, B, model, num_a=6, num_b=3): # <--- 修改 num_a 默认值
    """
    使用训练好的 Transformer 模型进行预测 (针对 A=6, B=3)。
    """
    try:
        # 特殊情况处理可以保留，但现在 A 的长度是 6
        if not set(A) & set(B) and len(B) == len(set(B)):
             print("快速路径：A、B无交集且B无重复，返回默认策略")
             # 返回的策略仍然是针对 B 中的 3 个元素
             return [[0, 1], [1, 1], [2, 1]], 0 # 默认位置 1 可能需要调整

        if not isinstance(model, TransformerMovePredictor):
             raise ValueError(f"需要 TransformerMovePredictor 实例, 但得到 {type(model)}")

        model.eval()

        # 检查输入长度是否为 6 和 3
        if len(A) != num_a or len(B) != num_b:
            raise ValueError(f"DNNpredict 期望输入 A 长度为 {num_a}, B 长度为 {num_b}, 但收到 A:{len(A)}, B:{len(B)}")

        input_sequence = np.array(A + B, dtype=np.float32)
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device) # (1, seq_len=9)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)
            # order_logits: (1, 3, 3), pos_preds: (1, 3)

        pred_order_indices = sample_order(order_logits).squeeze(0).cpu().numpy() # (3,)
        pred_moves_raw = pos_preds.squeeze(0).cpu().numpy() # (3,)

        # 位置裁剪的基准是初始 A 的长度，现在是 6
        pos_range_base = len(A) # = num_a = 6
        pred_moves_clipped = [int(np.clip(p, 0, pos_range_base + k)) for k, p in enumerate(pred_moves_raw)] # 位置范围 0 到 6+k

        best_moves = [[int(pred_order_indices[k]), pred_moves_clipped[k]] for k in range(num_b)] # num_b = 3

        final_order = [move[0] for move in best_moves]
        final_moves = [move[1] for move in best_moves]

        pred_score = calculate_score(A, B, final_order, final_moves)

        print(f"Transformer 预测 (6x3): A={A}, B={B} -> 策略={best_moves}, 预测得分={pred_score}")

        return best_moves, pred_score
    except Exception as e:
        print(f"使用 Transformer 预测时出错 (6x3): A={A}, B={B}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("返回默认策略")
        default_strategy = [[i, 1] for i in range(len(B))] # B 长度仍为 3
        default_score = calculate_score_by_strategy(A, B, default_strategy)
        return default_strategy, default_score


def train():
    """
    加载训练数据并启动 Transformer(6x3) 训练过程
    """
    try:
        with open(jsonfilename, "r") as f:
            train_data = json.load(f)
        print(f"成功加载训练数据，样本数: {len(train_data)}")
    except FileNotFoundError:
        print(f"错误: 未找到 JSON 文件 '{jsonfilename}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: JSON 文件 '{jsonfilename}' 格式无效")
        exit(1)
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        exit(1)

    if not train_data:
        print("错误: 没有加载到有效的训练数据")
        exit(1)

    # 定义固定的长度
    fixed_num_a = 6
    fixed_num_b = 3

    # 可选：预先过滤数据，只保留符合 6x3 长度的样本
    original_count = len(train_data)
    train_data = [s for s in train_data if len(s.get('A', [])) == fixed_num_a and len(s.get('B', [])) == fixed_num_b]
    filtered_count = len(train_data)
    if original_count > filtered_count:
        print(f"已过滤数据：保留了 {filtered_count} 个 A长度={fixed_num_a}, B长度={fixed_num_b} 的样本 (移除了 {original_count - filtered_count} 个)")

    if not train_data:
        print(f"错误: 过滤后没有找到符合长度 {fixed_num_a}x{fixed_num_b} 的训练数据")
        exit(1)


    # 调用 train_model 时传递固定长度，并使用新的模型路径
    train_model(train_data, epochs=300, batch_size=64 ,
                model_path="./trained/transformer_move_predictor_6x3.pth", # 新路径名
                num_a=fixed_num_a, num_b=fixed_num_b) # 传递 6 和 3


if __name__ == "__main__":
    train()

    # --- 可选的测试代码 ---
    # print("\n--- 开始测试预测功能 (6x3) ---")
    # try:
    #     num_a_test = 6 # <--- 修改
    #     num_b_test = 3
    #     # 确保这些参数与训练时一致
    #     d_model_test = 128
    #     nhead_test = 4
    #     num_layers_test = 3
    #     dim_ff_test = 256

    #     test_model = TransformerMovePredictor(
    #         num_a=num_a_test, num_b=num_b_test, d_model=d_model_test,
    #         nhead=nhead_test, num_encoder_layers=num_layers_test,
    #         dim_feedforward=dim_ff_test
    #     ).to(device)

    #     model_path_test = "./trained/transformer_move_predictor_6x3.pth" # <--- 修改
    #     test_model.load_state_dict(torch.load(model_path_test, map_location=device))
    #     print(f"成功加载模型: {model_path_test}")

    #     # 创建 6x3 的测试样本
    #     test_A1 = [1, 2, 3, 4, 5, 6]
    #     test_B1 = [7, 8, 9]

    #     test_A2 = [10, 30, 50, 20, 40, 60]
    #     test_B2 = [25, 35, 15]

    #     samples = [(test_A1, test_B1), (test_A2, test_B2)]

    #     for A, B in samples:
    #         predicted_strategy, predicted_score = DNNpredict(A, B, test_model, num_a=num_a_test, num_b=num_b_test) # <--- 传递 num_a=6
    #         print(f"输入: A={A}, B={B}")
    #         print(f"预测策略: {predicted_strategy}")
    #         print(f"预测得分: {predicted_score}")
    #         print("-" * 20)

    # except FileNotFoundError:
    #      print(f"测试错误: 找不到模型文件 '{model_path_test}'。请先运行训练。")
    # except Exception as e:
    #      print(f"测试预测时发生错误: {e}")
    #      import traceback
    #      traceback.print_exc()