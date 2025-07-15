import json
import math
import time
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
# 【新】导入匈牙利算法所需的库
from scipy.optimize import linear_sum_assignment

# 假设这个工具函数存在于您的项目中
from AI_algorithm.tool.tool import calculate_future_score_new

# --- 全局设置 (无变化) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA available, Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


# --- 1. 标准位置编码模块 (无变化) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
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


# --- 2. 双阶段集合处理器模型 (无变化) ---
class SetProcessorTransformer(nn.Module):
    """
    一个双阶段混合Transformer架构，实现了置换不变性与强大交互能力的结合。
    【修正】采用全局决策头，以匹配训练目标并提升性能。
    """

    def __init__(self, input_dim=6, d_model=256, nhead=4,
                 num_b_encoder_layers=2,
                 num_main_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(SetProcessorTransformer, self).__init__()
        # ... 其他初始化（num_a, num_b, d_model等）无变化 ...
        if num_a != 6 or num_b != 3:
            raise ValueError(f"期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        self.d_model = d_model

        # --- 通用模块 (无变化) ---
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_a + 5)

        # --- 阶段一: B-Set Processor (无变化) ---
        b_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.b_set_processor = nn.TransformerEncoder(b_encoder_layer, num_layers=num_b_encoder_layers)

        # --- 阶段二: Main Encoder (无变化) ---
        main_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.main_encoder = nn.TransformerEncoder(main_encoder_layer, num_layers=num_main_encoder_layers)

        # --- 【关键修正】全局决策头 ---
        # 输入维度是压平后的B组特征维度 (num_b * d_model)
        flat_b_dim = num_b * d_model

        # order_head 的输出维度是 num_b * num_b，之后会 reshape 成 (batch, num_b, num_b)
        # 代表了3个槽位，每个槽位对3张B牌的打分
        self.order_head = nn.Linear(flat_b_dim, num_b * num_b)

        # pos_head 的输出维度是 num_b，代表对3张B牌的位置预测
        self.pos_head = nn.Linear(flat_b_dim, num_b)

        self.init_weights()

    def init_weights(self):
        # ... init_weights 函数无变化 ...
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.input_embed.bias.data.zero_()
        self.type_embed.weight.data.uniform_(-initrange, initrange)
        self.order_head.weight.data.uniform_(-initrange, initrange)
        self.order_head.bias.data.zero_()
        self.pos_head.weight.data.uniform_(-initrange, initrange)
        self.pos_head.bias.data.zero_()

    def forward(self, src):
        # ... 前半部分（到memory生成）无变化 ...
        if src.dim() == 2:
            src = src.unsqueeze(0)

        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)
        embedded = input_embedded + type_embedded

        embedded_a = embedded[:, :self.num_a]
        embedded_b = embedded[:, self.num_a:]

        contextual_b_features = self.b_set_processor(embedded_b)
        encoded_a = self.pos_encoder(embedded_a)

        combined_sequence = torch.cat([encoded_a, contextual_b_features], dim=1)
        memory = self.main_encoder(combined_sequence)
        final_b_features = memory[:, self.num_a:]  # Shape: (batch, num_b, d_model)

        # --- 【关键修正】应用全局决策头 ---
        # 1. 压平B组特征，以获得全局信息
        b_features_flat = final_b_features.reshape(final_b_features.size(0), -1)  # Shape: (batch, num_b * d_model)

        # 2. 应用决策头
        order_logits_flat = self.order_head(b_features_flat)
        # 将压平的logits还原成 (batch, num_b, num_b) 的形状
        # 语义：order_logits[b, i, j] 代表第 i 个槽位选择第 j 张B牌的分数
        order_logits = order_logits_flat.view(-1, self.num_b, self.num_b)

        # pos_head 的输出直接就是对3张B牌的位置预测
        pos_preds = self.pos_head(b_features_flat)  # Shape: (batch, num_b)

        return order_logits, pos_preds


# --- 3. 数据准备函数 (无变化) ---
def prepare_data_for_new_model(sample: dict, num_a=6, num_b=3):
    A = sample["A"]
    B = sample["B"]
    best_moves = sample.get("best_moves", [])
    if len(A) != num_a or len(B) != num_b: return None
    B_counter = {val: B.count(val) for val in B};
    B_duplicates = sum(count - 1 for count in B_counter.values())
    intersection = set(A) & set(B);
    intersection_size = len(intersection)
    positions_in_A = {a_val: i for i, a_val in enumerate(A)};
    enhanced_sequence = []
    A_min, A_max, B_min, B_max = min(A), max(A), min(B), max(B)
    A_mean, A_std = np.mean(A), np.std(A)
    for i, val in enumerate(A):
        relative_position = i / num_a
        range_size = A_max - A_min
        min_max_scaled = (val - A_min) / range_size if range_size > 0 else 0.0
        is_extreme = 1.0 if (val == A_min or val == A_max) else 0.0
        z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
        enhanced_sequence.append(
            [val / sum(A), is_extreme, z_score, relative_position, min_max_scaled, intersection_size / num_a])
    for i, val in enumerate(B):
        is_in_A = 1.0 if val in A else 0.0
        position_in_A = positions_in_A.get(val, -1)
        relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1
        is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
        other_B = B[:i] + B[i + 1:]
        future_score = calculate_future_score_new(A, other_B)
        future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0
        enhanced_sequence.append([
            val / sum(B), future_score_ratio, is_in_A,
            relative_position_in_A, B_duplicates / num_b, is_extreme
        ])
    input_sequence = np.array(enhanced_sequence, dtype=np.float32)
    order_target = np.array([move[0] for move in best_moves], dtype=np.int64) if best_moves else None
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32) if best_moves else None
    if best_moves and len(set(order_target)) != num_b: return None
    if not best_moves: return input_sequence, None, None
    return input_sequence, order_target, pos_target


# --- 4. 训练函数 (无变化) ---
def train_new_model_with_monitoring(train_data, epochs=100, batch_size=2048,
                                    model_path="./trained/set_processor_transformer.pth", **kwargs):
    global stop_training
    stop_training = False
    model = SetProcessorTransformer(
        input_dim=6, d_model=256, nhead=4, num_b_encoder_layers=2,
        num_main_encoder_layers=3, dim_feedforward=512, dropout=0.1, num_a=6, num_b=3
    ).to(device)
    print("已初始化 SetProcessorTransformer 模型。")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get('lr_max', 0.0002))
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)
    model.train()
    best_loss = float('inf');
    patience_counter = 0;
    best_model_state = None
    patience = kwargs.get('patience', 10);
    min_delta = kwargs.get('min_delta', 0.01)
    try:
        for epoch in range(epochs):
            if stop_training: print("\n训练提前停止..."); break
            total_loss, batch_count = 0.0, 0
            np.random.shuffle(train_data)
            batch_list = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            progress_bar = tqdm(batch_list, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
            for batch in progress_bar:
                if stop_training: break
                inputs, order_targets, pos_targets = [], [], []
                for sample in batch:
                    proc_data = prepare_data_for_new_model(sample, num_a=6, num_b=3)
                    if proc_data and proc_data[0] is not None:
                        inputs.append(proc_data[0]);
                        order_targets.append(proc_data[1]);
                        pos_targets.append(proc_data[2])
                if not inputs: continue
                inputs_tensor = torch.FloatTensor(np.array(inputs)).to(device)
                order_targets_tensor = torch.LongTensor(np.array(order_targets)).to(device)
                pos_targets_tensor = torch.FloatTensor(np.array(pos_targets)).to(device)
                optimizer.zero_grad()
                order_logits, pos_preds = model(inputs_tensor)
                order_loss = order_criterion(order_logits, order_targets_tensor)
                pos_loss = pos_criterion(pos_preds, pos_targets_tensor)
                loss = order_loss + pos_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item();
                batch_count += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            progress_bar.close()
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs} 完成, 平均损失 (Avg Loss): {avg_loss:.4f}")
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss;
                patience_counter = 0;
                best_model_state = model.state_dict()
                torch.save(best_model_state, model_path)
                print(f"  -> 发现新的最佳模型，已保存至 {model_path}，损失值: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  -> 未发现显著改善，耐心计数: {patience_counter}/{patience}")
                if patience_counter >= patience: print(f"早停触发！连续 {patience} 个epoch无显著改善。"); break
    except KeyboardInterrupt:
        print("\n训练手动中断。")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}"); import traceback; traceback.print_exc()
    finally:
        if best_model_state is not None:
            print("\n加载训练过程中的最佳模型状态进行最终保存...")
            torch.save(best_model_state, model_path)
        print(f"训练结束。最终模型已保存在 {model_path}")


# --- 5. 【已修改】使用匈牙利算法的预测函数 ---
def predict_with_new_model(A, B, model, num_a=6, num_b=3):
    """
    使用新的SetProcessorTransformer模型进行预测。
    【修改】采用匈牙利算法对出牌顺序进行全局最优解码。
    """
    try:
        model.eval()
        data = prepare_data_for_new_model({"A": A, "B": B, "best_moves": []})
        if data is None or data[0] is None:
            raise ValueError("数据预处理失败")

        input_sequence, _, _ = data
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)  # 添加batch维度

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        # ----------- 【解码逻辑修改开始】 -----------

        # 1. 准备匈牙利算法的成本矩阵
        # order_logits shape: (1, 3, 3), 代表 (batch, B中牌的索引, 出牌槽位)
        # linear_sum_assignment 需要一个2D的numpy数组，并且是求最小值
        # 所以我们将logits取负，作为成本矩阵
        cost_matrix = -order_logits.squeeze(0).cpu().numpy()

        # 2. 调用匈牙利算法求解最优分配
        # row_ind 是B中牌的原始索引 (0, 1, 2)
        # col_ind 是分配给它们的出牌槽位 (e.g., [2, 0, 1])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. 构建最终的出牌顺序数组
        # final_order_indices 的索引是B中牌的原始索引，值是它被分配到的槽位
        # 例如 final_order_indices = [2, 0, 1] 意味着:
        # - B[0] -> 放到第2个槽位
        # - B[1] -> 放到第0个槽位
        # - B[2] -> 放到第1个槽位
        final_order_indices = np.zeros_like(row_ind)
        final_order_indices[row_ind] = col_ind

        # ----------- 【解码逻辑修改结束】 -----------

        # 4. 根据解码出的顺序，整理位置预测
        pred_moves_raw = pos_preds.squeeze(0).cpu().numpy()
        pos_range_base = len(A)

        # 5. 构建最终策略
        # final_strategy 的索引是出牌槽位(0, 1, 2)，值是[B中牌的原始索引, 预测位置]
        final_strategy = [[-1, -1] for _ in range(num_b)]
        for b_card_index, slot_index in enumerate(final_order_indices):
            # b_card_index: B中牌的原始索引 (0, 1, 2)
            # slot_index: 分配给这张牌的槽位 (0, 1, 2)

            # 获取这张牌的位置预测
            pos_pred = pred_moves_raw[b_card_index]

            # 裁剪位置
            clipped_pos = int(np.clip(pos_pred, 0, pos_range_base + slot_index))

            # 填入最终策略
            final_strategy[slot_index] = [b_card_index, clipped_pos]

        return final_strategy

    except Exception as e:
        print(f"使用新模型预测时出错: {e}")
        import traceback
        traceback.print_exc()
        return [[i, 1] for i in range(len(B))]


# --- 信号处理和主程序 (无变化) ---
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
    jsonfile_path = "json/data_Trans_fill.jsonl"
    new_model_path = "./trained/SetProcessor_Transformer.pth"

    try:
        train_data = load_jsonl_data(jsonfile_path)
        if train_data:
            train_new_model_with_monitoring(
                train_data,
                epochs=100,
                batch_size=2048,
                model_path=new_model_path,
                patience=5,
                lr_max=0.0002
            )
    except Exception as e:
        print(f"主程序出错: {e}")