import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from AI_algorithm.tool.tool import calculate_score_by_strategy
# 添加全局缓存


# from AI_algorithm.tool.tool import calculate_score_by_strategy
# 为了代码能独立运行，这里提供一个模拟的函数

def score_based_loss(order_logits, pos_preds, A_list, B_list, target_scores):
    """
    基于真实得分计算的损失函数

    参数:
    - order_logits: 形状为 [batch_size, num_b, num_b] 的顺序预测输出
    - pos_preds: 形状为 [batch_size, num_b] 的位置预测输出
    - A_list: 包含每个样本A序列的列表
    - B_list: 包含每个样本B序列的列表
    - target_scores: 每个样本的目标得分

    返回:
    - 基于得分的损失张量
    """
    batch_size = order_logits.size(0)
    batch_losses = []
    
    for i in range(batch_size):
        A = A_list[i]
        B = B_list[i]
        target_score = float(target_scores[i])  # 确保是浮点数
        
        order_logits_sample = order_logits[i].unsqueeze(0).detach()
        pred_order = sample_order(order_logits_sample).squeeze(0).cpu().detach().numpy()
        
        pos_preds_sample = pos_preds[i].cpu().detach().numpy()
        pred_moves = [int(np.clip(p, 0, len(A) + k)) for k, p in enumerate(pos_preds_sample)]
        
        pred_strategy = [[int(pred_order[j]), pred_moves[j]] for j in range(len(pred_order))]
        
        try:
            pred_score = float(calculate_score_by_strategy(A.copy(), B, pred_strategy))  # 确保是浮点数
            sample_loss = float((pred_score - target_score) ** 2)  # 确保是浮点数
        except Exception as e:
            print(f"计算得分时出错: {e}")
            sample_loss = 100.0
            
        batch_losses.append(sample_loss)
    
    # 直接使用FloatTensor，并且创建一个新的可导的连接点
    if batch_losses:
        loss_tensor = torch.FloatTensor(batch_losses).to(order_logits.device)
        # 创建一个新的计算节点以连接梯度
        return loss_tensor.mean() * (order_logits.sum() / order_logits.sum().detach())
    else:
        # 确保返回的零也是连接到计算图的
        return 0.0 * (order_logits.sum() / order_logits.sum().detach())

def score_based_loss_cuda(order_logits, pos_preds, A_tensor, B_tensor, target_scores):
    """
    基于得分的损失函数 - CUDA加速版本
    
    参数:
    - order_logits: [batch_size, num_b, num_b] 顺序预测
    - pos_preds: [batch_size, num_b] 位置预测
    - A_tensor: [batch_size, num_a] A序列，已转到GPU
    - B_tensor: [batch_size, num_b] B序列，已转到GPU
    - target_scores: [batch_size] 目标得分
    """
    batch_size = order_logits.size(0)
    
    # 使用softmax+argmax获取最佳顺序预测 (保留梯度连接)
    order_probs = F.softmax(order_logits, dim=2)
    _, pred_indices = torch.max(order_probs, dim=2)
    
    # 位置预测处理 (裁剪到合理范围)
    # 假设每个样本的A长度都是num_a
    num_a = A_tensor.size(1)
    num_b = B_tensor.size(1)
    
    # 创建位置范围上限张量 [num_b]
    pos_range = torch.arange(num_b, device=order_logits.device).float() + num_a
    # 将位置预测裁剪到合理范围 [batch_size, num_b]
    clipped_positions = torch.min(torch.max(pos_preds, torch.zeros_like(pos_preds)), pos_range)
    
    # 构建可微分的损失函数
    # 使用预测顺序和位置的概率加权平均
    pred_order_probs = torch.gather(order_probs, 2, pred_indices.unsqueeze(-1)).squeeze(-1)
    
    # 使用MSE替代直接计算score (作为近似)
    # 1. 顺序损失 - 监督信号
    order_mse = F.mse_loss(pred_order_probs, torch.ones_like(pred_order_probs))
    
    # 2. 位置损失 - 监督信号
    pos_mse = F.mse_loss(clipped_positions, pos_preds)
    
    return (order_mse + pos_mse).mean()

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

# 修改 TransformerMovePredictor 类的初始化，增加输入维度
class TransformerMovePredictor(nn.Module):
    # 修改默认参数以反映新的固定长度和增加的特征维度
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3): # 修改input_dim为6
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
# 修改数据预处理函数，添加更多特征
def prepare_data_transformer(sample: dict, num_a=6, num_b=3):
    """
    数据预处理：构建输入序列和目标输出，添加修改后的特征工程。
    """
    A = sample["A"]
    B = sample["B"]
    best_moves = sample["best_moves"]

    if len(A) != num_a or len(B) != num_b:
         print(f"Warning: Skipping sample with A={A}, B={B} due to unexpected length (expected A:{num_a}, B:{num_b}).")
         return None, None, None

    # 新增特征计算
    A_sum = sum(A)  # A序列的和
    B_sum = sum(B)  # B序列的和
    
    # 计算B序列中重复的元素数
    B_counter = {}
    for val in B:
        B_counter[val] = B_counter.get(val, 0) + 1
    B_duplicates = sum(count - 1 for count in B_counter.values())  # 重复的元素数
    
    # 交集特征
    intersection = set(A) & set(B)
    intersection_size = len(intersection)
    
    # 位置特征
    positions_in_A = {}
    for i, a_val in enumerate(A):
        positions_in_A[a_val] = i
    
    # 创建增强的输入序列
    enhanced_sequence = []
    
    # 处理A序列
    for i, val in enumerate(A):
        is_in_intersection = 1.0 if val in intersection else 0.0
        relative_position = i / num_a
        relative_size = val / A_sum if A_sum != 0 else 0  # 使用与总和的比例代替相对大小
        intersection_ratio = intersection_size / num_b
        
        enhanced_sequence.append([
            val,  # 原始值
            val / A_sum if A_sum != 0 else 0,  # 值占A总和的比例
            is_in_intersection,  # 是否在交集中
            relative_position,  # 相对位置
            relative_size,  # 相对大小（占总和的比例）
            intersection_ratio  # 交集大小比例
        ])
    
    # 处理B序列
    for i, val in enumerate(B):
        is_in_A = 1.0 if val in A else 0.0
        position_in_A = positions_in_A.get(val, -1)
        relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1
        relative_size = val / B_sum if B_sum != 0 else 0  # 使用与总和的比例
        
        enhanced_sequence.append([
            val,  # 原始值
            val / B_sum if B_sum != 0 else 0,  # 值占B总和的比例
            is_in_A,  # 是否在A中
            relative_position_in_A,  # 在A中的相对位置（如果存在）
            B_duplicates / num_b,  # B中重复元素的比例
            intersection_ratio  # 交集大小比例
        ])
    
    # 将序列转换为numpy数组
    input_sequence = np.array(enhanced_sequence, dtype=np.float32)

    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)

    if len(set(order_target)) != num_b:
        print(f"Warning: Skipping sample with invalid order target in best_moves: {order_target}")
        return None, None, None

    return input_sequence, order_target, pos_target

import signal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义全局变量
stop_training = False

def signal_handler(sig, frame):
    global stop_training
    print("\n检测到手动停止信号，准备保存当前模型...")
    stop_training = True

# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def lr_warmup(epoch, lr_max, warmup_epochs):
    if epoch < warmup_epochs:
        warmup_lr = lr_max * (epoch + 1) / warmup_epochs
        return warmup_lr
    else:
        return lr_max

# 修改训练函数中的模型初始化部分
def train_model(train_data, epochs=1000, batch_size=64, model_path="./trained/transformer_move_predictor_6x3.pth",
                num_a=6, num_b=3, warmup_epochs=50, lr_max=0.0001, lr_min=0.0000005):
    """
    训练 Transformer 模型 (针对 A=6, B=3)
    允许手动终止训练 (`Ctrl+C`) 并保存当前进度
    并加入学习率预热和余弦退火
    """
    global stop_training

    d_model = 256
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # 初始化模型，修改input_dim为6
    model = TransformerMovePredictor(input_dim=6, d_model=d_model, nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     dim_feedforward=dim_feedforward, dropout=dropout,
                                     num_a=num_a, num_b=num_b).to(device)

    conditional_print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    conditional_print("模型是否在GPU上:", next(model.parameters()).is_cuda)

    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)

    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    model.train()

    # 学习率预热 (Warmup) 和 余弦退火调度
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min)

    try:
        for epoch in range(epochs):
            if stop_training:
                break

            total_loss = 0.0
            batch_count = 0
            np.random.shuffle(train_data)

            for i in range(0, len(train_data), batch_size):
                if stop_training:
                    break

                batch = train_data[i:i + batch_size]
                inputs, order_targets, pos_targets = [], [], []
                
                for sample in batch:
                    input_seq, order_tgt, pos_tgt = prepare_data_transformer(sample, num_a=num_a, num_b=num_b)
                    if input_seq is None:
                        continue
                    inputs.append(input_seq)
                    order_targets.append(order_tgt)
                    pos_targets.append(pos_tgt)
                
                if not inputs:
                    continue

                inputs = torch.FloatTensor(np.array(inputs)).to(device)
                order_targets = torch.LongTensor(np.array(order_targets)).to(device)
                pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device)

                # 在训练前，根据学习率预热策略更新学习率
                if epoch < warmup_epochs:
                    new_lr = lr_warmup(epoch, lr_max, warmup_epochs)
                else:
                    # 余弦退火调度
                    new_lr = scheduler_cosine.get_lr()[0]  # 获取当前余弦退火后的学习率

                # 更新学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                optimizer.zero_grad()
                order_logits, pos_preds = model(inputs)

                # 计算损失
                order_loss = order_criterion(order_logits, order_targets)
                pos_loss = pos_criterion(pos_preds, pos_targets)

                # 计算基于得分的损失 (CUDA加速版本)
                score_loss = score_based_loss_cuda(order_logits, pos_preds, torch.tensor([sample["A"] for sample in batch], device=device), torch.tensor([sample["B"] for sample in batch], device=device), torch.tensor([sample.get("max_score", 60) for sample in batch], device=device))

                loss = order_loss + pos_loss + score_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            # 在每个epoch后更新余弦退火学习率
            if epoch >= warmup_epochs:
                scheduler_cosine.step()

            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {new_lr:.6f}")

    except KeyboardInterrupt:
        print("\n训练手动中断，正在保存当前模型...")

    finally:
        # 确保无论发生什么情况，模型都会被保存
        torch.save(model.state_dict(), model_path)
        print(f"Transformer 模型已保存至 {model_path}")

    return model

# 修改 DNNpredict 以使用新的固定长度
# 修改预测函数，使用相同的特征工程
def Transformer_predict(A, B, model, num_a=6, num_b=3):
    """
    使用训练好的 Transformer 模型进行预测 (针对 A=6, B=3)。
    """
    try:
        # 特殊情况处理可以保留，但现在 A 的长度是 6
        if not set(A) & set(B) and len(B) == len(set(B)):
             print("快速路径：A、B无交集且B无重复，返回默认策略")
             return [[0, 0], [1, 0], [2, 0]], 0

        if not isinstance(model, TransformerMovePredictor):
             raise ValueError(f"需要 TransformerMovePredictor 实例, 但得到 {type(model)}")

        model.eval()

        # 检查输入长度是否为 6 和 3
        if len(A) != num_a or len(B) != num_b:
            raise ValueError(f"DNNpredict 期望输入 A 长度为 {num_a}, B 长度为 {num_b}, 但收到 A:{len(A)}, B:{len(B)}")

        # 应用新的特征工程
        # 1. 计算序列和
        A_sum = sum(A)
        B_sum = sum(B)
        
        # 2. 计算B序列中重复的元素数
        B_counter = {}
        for val in B:
            B_counter[val] = B_counter.get(val, 0) + 1
        B_duplicates = sum(count - 1 for count in B_counter.values())
        
        # 3. 交集特征
        intersection = set(A) & set(B)
        intersection_size = len(intersection)
        
        # 4. 位置特征
        positions_in_A = {}
        for i, a_val in enumerate(A):
            positions_in_A[a_val] = i
        
        # 创建增强的输入序列
        enhanced_sequence = []
        
        # 处理A序列
        for i, val in enumerate(A):
            is_in_intersection = 1.0 if val in intersection else 0.0
            relative_position = i / num_a
            relative_size = val / A_sum if A_sum != 0 else 0
            intersection_ratio = intersection_size / num_b
            
            enhanced_sequence.append([
                val,  # 原始值
                val / A_sum if A_sum != 0 else 0,  # 值占A总和的比例
                is_in_intersection,  # 是否在交集中
                relative_position,  # 相对位置
                relative_size,  # 相对大小（占总和的比例）
                intersection_ratio  # 交集大小比例
            ])
        
        # 处理B序列
        for i, val in enumerate(B):
            is_in_A = 1.0 if val in A else 0.0
            position_in_A = positions_in_A.get(val, -1)
            relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1
            relative_size = val / B_sum if B_sum != 0 else 0
            
            enhanced_sequence.append([
                val,  # 原始值
                val / B_sum if B_sum != 0 else 0,  # 值占B总和的比例
                is_in_A,  # 是否在A中
                relative_position_in_A,  # 在A中的相对位置（如果存在）
                B_duplicates / num_b,  # B中重复元素的比例
                intersection_ratio  # 交集大小比例
            ])

        input_sequence = np.array(enhanced_sequence, dtype=np.float32)
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            order_logits, pos_preds = model(input_tensor)

        pred_order_indices = sample_order(order_logits).squeeze(0).cpu().numpy()
        pred_moves_raw = pos_preds.squeeze(0).cpu().numpy()

        pos_range_base = len(A)
        pred_moves_clipped = [int(np.clip(p, 0, pos_range_base + k)) for k, p in enumerate(pred_moves_raw)]

        best_moves = [[int(pred_order_indices[k]), pred_moves_clipped[k]] for k in range(num_b)]

        final_order = [move[0] for move in best_moves]
        final_moves = [move[1] for move in best_moves]

        pred_score = calculate_score(A, B, final_order, final_moves)

        # print(f"Transformer 预测 (6x3): A={A}, B={B} -> 策略={best_moves}, 预测得分={pred_score}")

        return best_moves, pred_score
    except Exception as e:
        print(f"使用 Transformer 预测时出错 (6x3): A={A}, B={B}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("返回默认策略")
        default_strategy = [[i, 1] for i in range(len(B))]
        default_score = calculate_score_by_strategy(A, B, default_strategy)
        return default_strategy, default_score

_json_cache = {}
def train():
    """
    加载训练数据并启动 Transformer(6x3) 训练过程
    """

    def load_json_data(json_file_path):
        """
        加载JSON数据并缓存
        """
        if json_file_path in _json_cache:
            return _json_cache[json_file_path]

        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
            _json_cache[json_file_path] = data
            print(f"成功加载训练数据，样本数: {len(data)}")
            return data
        except FileNotFoundError:
            print(f"错误: 未找到 JSON 文件 '{json_file_path}'")
            raise
        except json.JSONDecodeError:
            print(f"错误: JSON 文件 '{json_file_path}' 格式无效")
            raise
        except Exception as e:
            print(f"加载数据时发生未知错误: {e}")
            raise

    try:
        # 使用缓存加载数据
        train_data = load_json_data(jsonfilename)
    except Exception as e:
        print(f"加载数据失败: {e}")
        exit(1)

    if not train_data:
        print("错误: 没有加载到有效的训练数据")
        exit(1)

    # 定义固定的长度
    fixed_num_a = 6
    fixed_num_b = 3

    # # 可选：预先过滤数据，只保留符合 6x3 长度的样本
    # original_count = len(train_data)
    # train_data = [s for s in train_data if len(s.get('A', [])) == fixed_num_a and len(s.get('B', [])) == fixed_num_b]
    # filtered_count = len(train_data)
    # if original_count > filtered_count:
    #     print(f"已过滤数据：保留了 {filtered_count} 个 A长度={fixed_num_a}, B长度={fixed_num_b} 的样本 (移除了 {original_count - filtered_count} 个)")
    #
    # if not train_data:
    #     print(f"错误: 过滤后没有找到符合长度 {fixed_num_a}x{fixed_num_b} 的训练数据")
    #     exit(1)


    # 调用 train_model 时传递固定长度，并使用新的模型路径
    train_model(train_data, epochs=1000, batch_size=2048,
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

