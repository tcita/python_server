import json
import math
import time
import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from AI_algorithm.tool.tool import calculate_future_score



# --- 全局设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA available, Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


# --- 1. 选择性位置编码模块 ---
class SelectivePositionalEncoding(nn.Module):
    """
    只对序列的前 num_a 个元素（A部分）施加位置编码。
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(SelectivePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, num_a):
        """
        :param x: 输入张量, shape: (batch, seq_len, d_model)
        :param num_a: 序列中A部分的长度。
        """
        # 只对前 num_a 个位置（A的部分）加上位置编码
        x[:, :num_a, :] = x[:, :num_a, :] + self.pe[:, :num_a, :]
        return self.dropout(x)


# --- 2. 最终的混合架构模型 ---
class HybridTransformer(nn.Module):
    """
    一个混合Transformer架构：
    - 对A部分使用位置编码，将其视为序列。
    - 对B部分不使用位置编码，并用置换不变的头部处理，将其视为集合。
    """

    def __init__(self, input_dim=6, d_model=256, nhead=4, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(HybridTransformer, self).__init__()
        if num_a != 6 or num_b != 3:
            raise ValueError(f"期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        self.d_model = d_model
        seq_len = num_a + num_b

        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = SelectivePositionalEncoding(d_model, dropout, max_len=seq_len + 5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.order_head = nn.Linear(2 * d_model, num_b)
        self.pos_head = nn.Linear(2 * d_model, 1)

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

    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(0)

        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)

        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)

        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded, self.num_a)

        memory = self.transformer_encoder(embedded)

        b_features = memory[:, self.num_a:]
        b_global_feature = torch.mean(b_features, dim=1, keepdim=True)
        b_global_feature_expanded = b_global_feature.expand(-1, self.num_b, -1)
        combined_features = torch.cat([b_features, b_global_feature_expanded], dim=2)

        order_logits = self.order_head(combined_features)
        pos_preds = self.pos_head(combined_features).squeeze(-1)

        return order_logits, pos_preds


# --- 3. 最终版数据准备函数 ---
def prepare_data_hybrid_transformer(sample: dict, num_a=6, num_b=3):
    """
    为混合模型准备数据（最终修正版）。
    B的特征是置换不变的，其中“未来分”通过对“留一法”的子集排序来保证唯一性。
    """
    A = sample["A"]
    B = sample["B"]
    best_moves = sample.get("best_moves", []) # 使用 .get() 更安全

    if len(A) != num_a or len(B) != num_b:
        return None

    # --- 特征工程 (其他部分不变) ---
    B_counter = {val: B.count(val) for val in B}
    B_duplicates = sum(count - 1 for count in B_counter.values())
    intersection = set(A) & set(B)
    intersection_size = len(intersection)
    positions_in_A = {a_val: i for i, a_val in enumerate(A)}
    enhanced_sequence = []
    A_min, A_max, B_min, B_max = min(A), max(A), min(B), max(B)
    A_mean, A_std = np.mean(A), np.std(A)

    # 处理A序列 (保持不变)
    for i, val in enumerate(A):
        relative_position = i / num_a
        range_size = A_max - A_min
        min_max_scaled = (val - A_min) / range_size if range_size > 0 else 0.0
        is_extreme = 1.0 if (val == A_min or val == A_max) else 0.0
        z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
        enhanced_sequence.append(
            [val / sum(A), is_extreme, z_score, relative_position, min_max_scaled, intersection_size / num_a])

    # 处理B序列 (使用最终修正的“未来分”逻辑)
    for i, val in enumerate(B):
        is_in_A = 1.0 if val in A else 0.0
        position_in_A = positions_in_A.get(val, -1)
        relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1
        is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
        other_B = B[:i] + B[i + 1:]
        sorted_other_B = sorted(other_B)
        # TODO:修正future_score
        future_score = calculate_future_score(A, sorted_other_B)
        future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0
        enhanced_sequence.append([
            val / sum(B), future_score_ratio, is_in_A,
            relative_position_in_A, B_duplicates / num_b, is_extreme
        ])

    input_sequence = np.array(enhanced_sequence, dtype=np.float32)
    order_target = np.array([move[0] for move in best_moves], dtype=np.int64) if best_moves else None
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32) if best_moves else None


    # 仅在训练时（即 best_moves 存在时）检查 order_target
    if best_moves and len(set(order_target)) != num_b:
        return None # 在训练时，如果目标无效，则跳过该样本

    # 预测时，只返回 input_sequence
    if not best_moves:
        return input_sequence, None, None

    return input_sequence, order_target, pos_target
# --- 4. 训练函数 ---
from tqdm import tqdm  # 确保在文件顶部导入 tqdm


def train_model_with_monitoring(train_data, epochs=100, batch_size=2048, model_path="./trained/hybrid_transformer.pth",
                                **kwargs):
    """
    使用tqdm进度条来监控训练过程的最终版训练函数。
    """
    # --- 全局设置与初始化 ---
    global stop_training
    stop_training = False  # 确保每次调用都重置停止标志

    d_model = 256
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    num_a = 6
    num_b = 3

    # 初始化我们最终的混合模型
    model = HybridTransformer(
        input_dim=6, d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        num_a=num_a, num_b=num_b
    ).to(device)

    print("已初始化 HybridTransformer 模型。")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get('lr_max', 0.0002))
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)
    model.train()

    # 早停机制相关变量
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    patience = kwargs.get('patience', 10)
    min_delta = kwargs.get('min_delta', 0.01)

    # --- 训练主循环 ---
    try:
        for epoch in range(epochs):
            if stop_training:
                print("\n训练提前停止...")
                break

            total_loss, batch_count = 0.0, 0
            np.random.shuffle(train_data)

            # 使用tqdm创建进度条
            batch_list = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            progress_bar = tqdm(batch_list, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)

            for batch in progress_bar:
                if stop_training: break

                inputs, order_targets, pos_targets = [], [], []
                for sample in batch:
                    # 调用最终版的数据准备函数
                    proc_data = prepare_data_hybrid_transformer(sample, num_a=num_a, num_b=num_b)
                    if proc_data:
                        inputs.append(proc_data[0])
                        order_targets.append(proc_data[1])
                        pos_targets.append(proc_data[2])

                if not inputs: continue

                inputs_tensor = torch.FloatTensor(np.array(inputs)).to(device)
                order_targets_tensor = torch.LongTensor(np.array(order_targets)).to(device)
                pos_targets_tensor = torch.FloatTensor(np.array(pos_targets)).to(device)

                optimizer.zero_grad()
                order_logits, pos_preds = model(inputs_tensor)

                # 正确计算损失
                order_logits_permuted = order_logits.permute(0, 2, 1)
                order_loss = order_criterion(order_logits_permuted, order_targets_tensor)
                pos_loss = pos_criterion(pos_preds, pos_targets_tensor)
                loss = order_loss + pos_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 更新tqdm进度条的后缀，实时显示损失
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            progress_bar.close()  # 关闭当前epoch的进度条

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs} 完成, 平均损失 (Avg Loss): {avg_loss:.4f}")

            # 早停检查
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # 保存当前最佳模型状态到内存和文件
                best_model_state = model.state_dict()
                torch.save(best_model_state, model_path)
                print(f"  -> 发现新的最佳模型，已保存至 {model_path}，损失值: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  -> 未发现显著改善，耐心计数: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"早停触发！连续 {patience} 个epoch无显著改善。")
                    break

    except KeyboardInterrupt:
        print("\n训练手动中断。")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在任何情况下，如果存在最佳模型状态，我们都用它来保存最终模型
        if best_model_state is not None:
            print("\n加载训练过程中的最佳模型状态进行最终保存...")
            torch.save(best_model_state, model_path)

        print(f"训练结束。最终模型已保存在 {model_path}")
# --- 5. 与新模型匹配的预测函数 ---
def Transformer_predict_hybrid(A, B, model, num_a=6, num_b=3):
    """
    使用训练好的混合模型进行预测。
    """
    try:
        model.eval()

        # 特征工程 (与 prepare_data_hybrid_transformer 保持一致)
        data = prepare_data_hybrid_transformer({"A": A, "B": B, "best_moves": []})
        if data is None:
            raise ValueError("数据预处理失败")

        input_sequence, _, _ = data
        input_tensor = torch.FloatTensor(input_sequence).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        # 解码逻辑
        final_order_indices = torch.argmax(order_logits.permute(0, 2, 1), dim=2).squeeze(0).cpu().numpy()
        pred_moves_raw = pos_preds.squeeze(0).cpu().numpy()

        reordered_pos_raw = pred_moves_raw[final_order_indices]
        pos_range_base = len(A)
        pred_moves_clipped = [int(np.clip(p, 0, pos_range_base + k)) for k, p in enumerate(reordered_pos_raw)]
        best_moves = [[int(final_order_indices[k]), pred_moves_clipped[k]] for k in range(num_b)]

        return best_moves
    except Exception as e:
        print(f"使用 Hybrid-Transformer 预测时出错: {e}")
        return [[i, 1] for i in range(len(B))]


# --- 信号处理和主程序 ---
stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True


signal.signal(signal.SIGINT, signal_handler)

_json_cache = {}


def load_jsonl_data(json_file_path):
    if json_file_path in _json_cache: return _json_cache[json_file_path]
    try:
        with open(json_file_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
        _json_cache[json_file_path] = data
        print(f"成功加载训练数据，样本数: {len(data)}")
        return data
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        raise


if __name__ == "__main__":
    # --- 训练新的混合模型 ---
    jsonfile_path = "json/data_Trans_fill.jsonl"
    # 为新模型指定一个清晰的路径
    hybrid_model_path = "./trained/Hybrid_Transformer.pth"

    try:
        train_data = load_jsonl_data(jsonfile_path)
        if train_data:
            train_model_with_monitoring(
                train_data,
                epochs=100,
                batch_size=2048,
                model_path=hybrid_model_path,
                patience=5,
                lr_max=0.0002
            )
    except Exception as e:
        print(f"主程序出错: {e}")