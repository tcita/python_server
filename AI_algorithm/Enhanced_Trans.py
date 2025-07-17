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
    【修正】采用基于查询的决策头，以实现真正的置换不变性。
    """

    def __init__(self, input_dim=6, d_model=256, nhead=4,
                 num_b_encoder_layers=2,
                 num_main_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(SetProcessorTransformer, self).__init__()
        if num_a != 6 or num_b != 3:
            raise ValueError(f"期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        self.d_model = d_model

        # --- 通用模块 ---
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_a + 5)

        # --- 阶段一: B-Set Processor ---
        b_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.b_set_processor = nn.TransformerEncoder(b_encoder_layer, num_layers=num_b_encoder_layers)

        # --- 阶段二: Main Encoder ---
        main_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.main_encoder = nn.TransformerEncoder(main_encoder_layer, num_layers=num_main_encoder_layers)

        # --- 【正确方案】基于查询的决策头 ---
        # 1. 引入可学习的 "槽位查询" 参数。
        self.slot_queries = nn.Parameter(torch.randn(1, num_b, d_model))

        # 2. 位置预测头 (pos_head)，作用于每个B牌的特征。
        self.pos_head = nn.Linear(d_model, 1)

        # 初始化所有权重
        self.init_weights()

    def init_weights(self):
        """
        【已修正】初始化权重，移除对旧order_head的引用，增加对slot_queries的初始化。
        """
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.input_embed.bias.data.zero_()
        self.type_embed.weight.data.uniform_(-initrange, initrange)

        # 初始化新的 slot_queries 参数
        self.slot_queries.data.uniform_(-initrange, initrange)

        # 初始化 pos_head
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

        embedded_a = embedded[:, :self.num_a]
        embedded_b = embedded[:, self.num_a:]

        contextual_b_features = self.b_set_processor(embedded_b)
        encoded_a = self.pos_encoder(embedded_a)

        combined_sequence = torch.cat([encoded_a, contextual_b_features], dim=1)
        memory = self.main_encoder(combined_sequence)
        final_b_features = memory[:, self.num_a:]  # Shape: (batch, num_b, d_model)

        # --- 应用基于查询的决策头 ---
        # 1. 计算出牌顺序 (Order Prediction)
        slot_q = self.slot_queries.expand(final_b_features.size(0), -1, -1)
        order_logits = torch.bmm(slot_q, final_b_features.transpose(1, 2))

        # 2. 计算出牌位置 (Position Prediction)
        pos_preds_per_card = self.pos_head(final_b_features)
        pos_preds = pos_preds_per_card.squeeze(-1)

        return order_logits, pos_preds

# --- 3. 数据准备函数 (无变化) ---
def prepare_data_for_enhanced_Transformer(sample: dict, num_a=6, num_b=3):
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
        # is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
        has_duplicates_in_b = 1.0 if len(B) != len(set(B)) else 0.0
        other_B = B[:i] + B[i + 1:]
        future_score = calculate_future_score_new(A, other_B)
        future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0
        enhanced_sequence.append([
            val / sum(B), future_score_ratio, is_in_A,
            relative_position_in_A, B_duplicates / num_b, has_duplicates_in_b
        ])
    input_sequence = np.array(enhanced_sequence, dtype=np.float32)
    order_target = np.array([move[0] for move in best_moves], dtype=np.int64) if best_moves else None
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32) if best_moves else None
    if best_moves and len(set(order_target)) != num_b: return None
    if not best_moves: return input_sequence, None, None
    return input_sequence, order_target, pos_target


# --- 4. 训练函数 (无变化) ---
def train_enhanced_Transformer_with_monitoring(train_data, epochs=100, batch_size=2048,
                                               model_path="./trained/set_processor_transformer.pth", **kwargs):
    """
    【重写版本】
    修正了位置损失(pos_loss)的计算，确保预测与目标的张量对齐。
    """
    global stop_training
    stop_training = False
    model = SetProcessorTransformer(
        input_dim=6, d_model=256, nhead=4, num_b_encoder_layers=2,
        num_main_encoder_layers=3, dim_feedforward=512, dropout=0.1, num_a=6, num_b=3
    ).to(device)
    print("已初始化 SetProcessorTransformer 模型。")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get('lr_max', 0.00005))
    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    patience = kwargs.get('patience', 10)
    min_delta = kwargs.get('min_delta', 0.01)

    try:
        for epoch in range(epochs):
            if stop_training:
                print("\n训练提前停止...")
                break

            total_loss, batch_count = 0.0, 0
            np.random.shuffle(train_data)
            batch_list = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            progress_bar = tqdm(batch_list, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)

            for batch in progress_bar:
                if stop_training:
                    break

                inputs, order_targets, pos_targets = [], [], []
                for sample in batch:
                    proc_data = prepare_data_for_enhanced_Transformer(sample, num_a=6, num_b=3)
                    if proc_data and proc_data[1] is not None and proc_data[2] is not None:
                        inputs.append(proc_data[0])
                        order_targets.append(proc_data[1])
                        pos_targets.append(proc_data[2])

                if not inputs:
                    continue

                inputs_tensor = torch.FloatTensor(np.array(inputs)).to(device)
                order_targets_tensor = torch.LongTensor(np.array(order_targets)).to(device)
                pos_targets_tensor = torch.FloatTensor(np.array(pos_targets)).to(device)

                optimizer.zero_grad()
                order_logits, pos_preds = model(inputs_tensor)

                # 1. 顺序损失 (order_loss) - 计算方式正确
                order_loss = order_criterion(order_logits, order_targets_tensor)

                # 2. 【关键修正】位置损失 (pos_loss)
                # pos_preds 是按 "输入B牌" 顺序排列的预测: [pred_pos(B0), pred_pos(B1), pred_pos(B2)]
                # pos_targets_tensor 是按 "出牌槽位" 顺序排列的目标: [target_pos(Slot0), target_pos(Slot1), target_pos(Slot2)]
                # order_targets_tensor 记录了槽位到牌的映射: [card_idx_for_Slot0, card_idx_for_Slot1, ...]
                # 我们使用 gather, 按照槽位顺序从 pos_preds 中提取正确的牌的预测，使其与目标对齐。
                reordered_pos_preds = pos_preds.gather(1, order_targets_tensor)
                pos_loss = pos_criterion(reordered_pos_preds, pos_targets_tensor)

                # loss = order_loss + pos_loss

                # 定义损失权重

                weight_order = 10.0
                weight_pos = 1.0

                loss = (weight_order * order_loss) + (weight_pos * pos_loss)  # <--- 使用加权损失
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", order_loss=f"{order_loss.item():.4f}",
                                         pos_loss=f"{pos_loss.item():.4f}")

            progress_bar.close()
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs} 完成, 平均损失 (Avg Loss): {avg_loss:.4f}")

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
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
        if best_model_state is not None:
            print("\n加载训练过程中的最佳模型状态进行最终保存...")
            torch.save(best_model_state, model_path)
        print(f"训练结束。最终模型已保存在 {model_path}")

# --- 5. 【已修改】使用匈牙利算法的预测函数 ---
def predict_with_enhanced_Transformer(A, B, model, num_a=6, num_b=3):
    """
    【重写版本】
    修正了解码逻辑，确保使用正确的索引来匹配牌的预测位置和最终的出牌槽位。
    """
    try:
        model.eval()
        data = prepare_data_for_enhanced_Transformer({"A": A, "B": B, "best_moves": []})
        if data is None or data[0] is None:
            raise ValueError("数据预处理失败")

        input_sequence, _, _ = data
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        # --- 【关键修正】解码逻辑 ---

        # 1. 准备匈牙利算法的成本矩阵
        # cost_matrix[i, j] 代表 "槽位 i" 选择 "牌 j" 的成本 (负的logits)
        cost_matrix = -order_logits.squeeze(0).cpu().numpy()

        # 2. 调用匈牙利算法求解最优分配
        # slot_indices_optimal 是槽位索引 (值为 0, 1, 2)
        # card_indices_optimal 是对应的被选中的牌的索引
        slot_indices_optimal, card_indices_optimal = linear_sum_assignment(cost_matrix)

        # 3. 创建从槽位到牌索引的清晰映射
        # card_assignment_per_slot[i] 的值是应该被放入 "槽位 i" 的 "牌的索引"
        card_assignment_per_slot = np.zeros_like(slot_indices_optimal)
        card_assignment_per_slot[slot_indices_optimal] = card_indices_optimal

        # 4. 获取按 "输入B牌" 顺序排列的位置预测
        pred_moves_raw = pos_preds.squeeze(0).cpu().numpy()
        pos_range_base = len(A)

        # 5. 构建最终策略
        final_strategy = [[-1, -1] for _ in range(num_b)]

        # 遍历每个槽位 (0, 1, 2)
        for slot_idx, card_idx in enumerate(card_assignment_per_slot):
            # slot_idx: 当前槽位的索引 (0, 1, 2)
            # card_idx: 分配给这个槽位的B牌的索引

            # 使用 "牌的索引" (card_idx) 从原始预测中获取对应牌的位置预测
            pos_pred = pred_moves_raw[card_idx]

            # 裁剪位置，范围与当前 "槽位" (slot_idx) 相关
            clipped_pos = int(np.clip(pos_pred, 0, pos_range_base + slot_idx))

            # 将 [牌的索引, 预测位置] 放入正确的 "槽位" (slot_idx)
            final_strategy[slot_idx] = [card_idx, clipped_pos]

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
            train_enhanced_Transformer_with_monitoring(
                train_data,
                epochs=100,
                batch_size=2048,
                model_path=new_model_path,
                patience=3,
                lr_max=0.00003
            )
    except Exception as e:
        print(f"主程序出错: {e}")