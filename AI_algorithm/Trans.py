import json
import math
import time

import torch
import torch.nn as nn

from AI_algorithm.GA import GA_Strategy
from AI_algorithm.brute_force import recursive_Strategy
from AI_algorithm.tool.tool import calculate_score_by_strategy, calculate_future_score, load_best_genome, \
    deal_cards_tool, strategy_TrueScore



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"CUDA available，Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

#
# def calculate_score(A, B, order, moves):
#     strategy = [[int(order[i]), int(moves[i])] for i in range(len(order))]
#     total_score = calculate_score_by_strategy(A, B, strategy)
#     return total_score


# --- Transformer 相关组件 ---

class PositionalEncoding(nn.Module):

    """
        __init__ 方法初始化对象的位置编码模块。

        d_model: 表示模型的维度（即嵌入向量的维度）。
        dropout: Dropout 的概率，默认为 0.1。 训练时随机丢弃10%的神经元 ，用于防止过拟合。
        max_len: 最大序列长度，表示可以支持的最大位置数

    """
    def __init__(self, d_model, dropout=0.1, max_len=20):  # 定义一个足够大的 max_len 以容纳预期的最长序列
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 形状为[max_len d_model]的 0张量
        pe = torch.zeros(max_len, d_model)

        # 创建一个从 0 到 max_len-1 的一维张量，并将其形状从 [max_len] 变为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)



        # div_term = =[10000^(0/-d_model), 10000^(2/-d_model), ..., 10000^((d_model-2)/-d_model)]
        # d_model/2个元素

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # position * div_term得到形状为 [max_len, d_model/2]的张量

        # position * div_term 矩阵的结构：
        # [ [i / 10000^(0/d_model)]^T,    [i / 10000^(2/d_model)]^T  ,...,   [i / 10000^((d_model-2)/d_model)]^T ]

        # 其中 i ∈ [0, 1, 2, ..., max_len-1]

        # 对于position * div_term 这个矩阵而言,元素和行,列的映射关系是 f(i,j)=i/k   k=10000^ (2j / d_model  )
        # 在第j列中,j为参数,i为自变量 ,随着行索引i的增加,元素值从最小值0开始向(max_len-1)/k线性增长
        # 在第i行中,i为参数,j为自变量 随着列索引j增加，因为f(i,j)中的变量j出现在了分母中一个指数项的指数位置上 ,所以元素值从最大值i开始向0呈现指数级衰减


        # 令ω_j=1/10000^ (2j / d_model  ),θ_j=kω_j,k为相对position的位置
        #
        # 由于 旋转矩阵M(θ_j)^T*[sin(position*ω_j)   cos(position*ω_j)]^T= [sin(position+k)*ω_j  cos(position+k)*ω_j]
        # 这样的设计便于获取相对position偏移量为k的位置的位置编码 (通过对[pe[2i] pe[2i+1]]^T进行线性变换得到)
        # 因此完整的位置编码pe[position+k]可以通过旋转矩阵组成的分块对角矩阵 M_k  与pe(position)做矩阵乘法得到


        # 将position * div_term矩阵依据列序号的偶数和奇数分为两种情况,并分别取正余弦,在不拆分该矩阵的情况下来初始化位置编码矩阵

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第0维增加一个batch维度，将形状从[max_len, d_model]变为[1, max_len, d_model],以便与词嵌入向量相加
        pe = pe.unsqueeze(0)
        #使用 register_buffer 将 pe 注册为模型的缓冲区。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 向量x逐元素加上位置编码的当前输入长度切片,广播机制会调整张量的形状使得能够相加
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class TransformerMovePredictor(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(TransformerMovePredictor, self).__init__()
        # 验证传入的 num_a 和 num_b 是否符合预期
        if num_a != 6 or num_b != 3:
            raise ValueError(
                f"TransformerMovePredictor 初始化期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        #  seq_len=6 + 3 = 9
        self.seq_len = num_a + num_b
        self.d_model = d_model

        # 定义线性投影层 (y = xW^T + b)，Linear将输入in_features映射到out_features维度。
        # 初始化各自独立的权重矩阵 W 和偏置向量 b

        # # 将原始输入特征 (input_dim) 映射到模型的隐藏维度 (d_model)
        self.input_embed = nn.Linear(input_dim, d_model)

        # # 顺序预测头：预测它们两两之间的相对顺序关系 (输出一个 num_b x num_b 的关系矩阵)
        self.order_head = nn.Linear(num_b * d_model, num_b * num_b)

        # 位置预测头：为每个元素预测其绝对位置或类别 (输出 num_b 个值)
        self.pos_head = nn.Linear(num_b * d_model, num_b)

        # 定义类别嵌入层，创建查找表,用于区分两种不同类型(输入来自A或者B)的输入
        self.type_embed = nn.Embedding(2, d_model)
        # 初始化位置编码模块 max_len用于缓冲
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len + 5)

        # 定义编码器layer和编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)



        # 初始化模型中的参数
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # 初始化各层权重与偏置
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.input_embed.bias.data.zero_()
        self.type_embed.weight.data.uniform_(-initrange, initrange)
        self.order_head.weight.data.uniform_(-initrange, initrange)
        self.order_head.bias.data.zero_()
        self.pos_head.weight.data.uniform_(-initrange, initrange)
        self.pos_head.bias.data.zero_()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 输入数据标准化
        if src.dim() == 2:
            src = src.unsqueeze(-1)  # (批处理大小, 每个样本的长度, 1)
        # 词嵌入.将输入的词ID序列 (src) 转换为词义向量 (self.input_embed)。将这些词义向量放大合适的比例，使得适合在后续与位置信息向量相加
        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)

        # 创建num_a 个 0 组成的类型向量 和 num_b 个 1 组成的类型向量,并将它们拼接,并填入类型查找表(type_embed)中
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)  # (batch_size, seq_len)
        type_embedded = self.type_embed(type_ids)

        # 逐元素相加词嵌入和类型嵌入形成嵌入向量,这个向量会作为PositionalEncoding中的forward的参数x,来调用forward
        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded)

        memory = self.transformer_encoder(embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # memory shape: (batch_size, seq_len=9, d_model)

        # 提取 B 对应的输出特征 (最后 num_b=3 个)
        # 使用 self.num_a (现在是6) 作为索引起点
        b_features = memory[:, self.num_a:]  # (batch_size, num_b=3, d_model)

        b_features_flat = b_features.reshape(b_features.size(0), -1)  # (batch_size, num_b * d_model)

        order_logits_flat = self.order_head(b_features_flat)
        order_logits = order_logits_flat.view(-1, self.num_b, self.num_b)  # (batch_size, 3, 3)

        pos_preds = self.pos_head(b_features_flat)  # (batch_size, 3)

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
    num_b = order.shape[1]  # Should be 3
    valid_permutations = torch.stack([torch.randperm(num_b) for _ in range(batch_size)]).to(order.device)

    is_invalid = torch.zeros(batch_size, dtype=torch.bool, device=order.device)
    for b in range(batch_size):
        if len(torch.unique(order[b])) != num_b:
            is_invalid[b] = True

    order[is_invalid] = valid_permutations[is_invalid]
    return order



def prepare_data_transformer(sample: dict, num_a=6, num_b=3):
    """
    数据预处理：构建输入序列和目标输出，添加特征工程。
    """


    A = sample["A"]
    B = sample["B"]
    best_moves = sample["best_moves"]

    if len(A) != num_a or len(B) != num_b:
        print(f"Warning: Skipping sample with A={A}, B={B} due to unexpected length (expected A:{num_a}, B:{num_b}).")
        return None, None, None





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

    A_min = min(A)
    A_max = max(A)
    B_min = min(B)
    B_max = max(B)
    A_mean = np.mean(A)
    A_std = np.std(A)
    # B_mean = np.mean(B)
    # B_std = np.std(B)
    # 处理A序列
    for i, val in enumerate(A):

        relative_position = i / num_a

        # z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
        range_size = A_max - A_min
        min_max_scaled = (val - A_min) / range_size if range_size > 0 else 0.0
        is_extreme = 1.0 if (val == A_min or val == A_max) else 0.0
        z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
        enhanced_sequence.append([
            val/sum(A),  # 本身的值占A的元素总和的比例
            is_extreme,  # 是否是极值
            z_score,#该值与均值之间的距离
            relative_position,  # 该元素在A中的相对位置
            min_max_scaled,  #将一个值等比例地映射到 [0, 1] 区间
            intersection_size/num_a  # A与B交集大小
        ])

    # 处理B序列
    for i, val in enumerate(B):
        is_in_A = 1.0 if val in A else 0.0
        position_in_A = positions_in_A.get(val, -1)
        relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1

        is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
        # z_score = (val - B_mean) / B_std if B_std > 0 else 0.0
        # 计算未来得分特征
        # 假设当前B[i]已经被处理，计算剩余B的未来得分
        remaining_B = B[i + 1:] if i < len(B) - 1 else []
        future_score = calculate_future_score(A, remaining_B)
        future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0

        enhanced_sequence.append([
            val/sum(B),  # 本身的值占B的元素总和的比例

            future_score_ratio,  # 未来得分占可能的最大得分(全部消除)的占比

            is_in_A,  # 是否在A中
            relative_position_in_A,  # 该值在A中的相对位置（不存在为-0.1）
            B_duplicates / num_b,  # B中重复元素占所有B元素数的比例

            is_extreme  # 是否是极值
        ])

    # 将序列转换为numpy数组
    input_sequence = np.array(enhanced_sequence, dtype=np.float32)

    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)

    if len(set(order_target)) != num_b:
        print(f"Warning: Skipping sample with invalid order target in best_moves: {order_target}")
        return None, None, None


    result = (input_sequence, order_target, pos_target)


    return result


import signal

# 定义全局变量
stop_training = False


def signal_handler(sig, frame):
    global stop_training
    # print("\n检测到手动停止信号，准备保存当前模型...")
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


def train_model(train_data, epochs=300, batch_size=64, model_path="./trained/transformer_move_predictor_6x3.pth",
                num_a=6, num_b=3, warmup_epochs=10, lr_max=0.0001, lr_min=0.0000005,
                patience=10, min_delta=0.01):
    # 添加GPU信息检查
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("警告: 未检测到可用的GPU!")

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

    print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("模型是否在GPU上:", next(model.parameters()).is_cuda)

    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)

    order_criterion = nn.CrossEntropyLoss().to(device)
    pos_criterion = nn.MSELoss().to(device)

    model.train()

    # 学习率预热 (Warmup) 和 余弦退火调度
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min)

    # 早停相关变量
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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

                loss = order_loss + pos_loss

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 每个batch的训练日志
                if batch_count % 10 == 0:  # 每10个batch打印一次
                    print(f"    Batch {batch_count}/{len(train_data)//batch_size}, Loss: {loss.item():.4f},LR: {new_lr:.7f}")

            # 在每个epoch后更新余弦退火学习率
            if epoch >= warmup_epochs:
                scheduler_cosine.step()

            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {new_lr:.7f}")

            # 早停检查
            if avg_loss < best_loss - min_delta:  # 有显著改善
                best_loss = avg_loss
                patience_counter = 0
                # 保存当前最佳模型状态
                best_model_state = model.state_dict().copy()
                print(f"发现新的最佳模型，损失值: {best_loss:.4f}")
            else:  # 没有显著改善
                patience_counter += 1
                print(f"未发现显著改善，耐心计数: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"早停触发！连续 {patience} 个epoch无显著改善")
                    # 恢复到最佳模型状态
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    except KeyboardInterrupt:
        print("\n训练手动中断，正在保存当前模型...")

    except Exception as e:
        print(f"\n训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # 确保无论发生什么情况，模型都会被保存
        try:
            # 如果触发了早停并有最佳模型，确保我们使用最佳模型
            if patience_counter >= patience and best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("已加载训练过程中的最佳模型状态")

            torch.save(model.state_dict(), model_path)
            print(f"Transformer 模型已保存至 {model_path}")
        except Exception as save_error:
            print(f"保存模型时发生错误: {str(save_error)}")

    return model


# 修改预测函数，使用相同的特征工程
# 添加预测缓存
_prediction_cache = {}


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
            raise ValueError(f" 期望输入 A 长度为 {num_a}, B 长度为 {num_b}, 但收到 A:{len(A)}, B:{len(B)}")

       # 特征计算  与prepare_data_transformer函数相同
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

        A_min = min(A)
        A_max = max(A)
        B_min = min(B)
        B_max = max(B)
        A_mean = np.mean(A)
        A_std = np.std(A)
        # B_mean = np.mean(B)
        # B_std = np.std(B)
        # 处理A序列
        for i, val in enumerate(A):
            relative_position = i / num_a

            # z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
            range_size = A_max - A_min
            min_max_scaled = (val - A_min) / range_size if range_size > 0 else 0.0
            is_extreme = 1.0 if (val == A_min or val == A_max) else 0.0
            z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
            enhanced_sequence.append([
                val / sum(A),  # 本身的值占A的元素总和的比例
                is_extreme,  # 是否是极值
                z_score,  # 该值与均值之间的距离
                relative_position,  # 该元素在A中的相对位置
                min_max_scaled,  # 将一个值的原始位置，等比例地映射到 [0, 1] 区间
                intersection_size/num_a  # A与B交集大小占A元素数的比例
            ])

        # 处理B序列
        for i, val in enumerate(B):
            is_in_A = 1.0 if val in A else 0.0
            position_in_A = positions_in_A.get(val, -1)
            relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1

            is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
            # z_score = (val - B_mean) / B_std if B_std > 0 else 0.0
            # 计算未来得分特征
            # 假设当前B[i]已经被处理，计算剩余B的未来得分
            remaining_B = B[i + 1:] if i < len(B) - 1 else []
            future_score = calculate_future_score(A, remaining_B)
            future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0

            enhanced_sequence.append([
                val / sum(B),  # 本身的值占B的元素总和的比例

                future_score_ratio,  # 未来得分占可能的最大得分(全部消除)的占比

                is_in_A,  # 是否在A中
                relative_position_in_A,  # 该值在A中的相对位置（不存在为-0.1）
                B_duplicates / num_b,  # B中重复元素占所有B元素数的比例

                is_extreme  # 是否是极值
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

        # final_order = [move[0] for move in best_moves]
        # final_moves = [move[1] for move in best_moves]

        # pred_score = calculate_score(A, B, final_order, final_moves)

        # print(f"Transformer 预测 (6x3): A={A}, B={B} -> 策略={best_moves}, 预测得分={pred_score}")




        return best_moves
    except Exception as e:
        print(f"使用 Transformer 预测时出错 (6x3): A={A}, B={B}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("返回默认策略")
        default_strategy = [[i, 1] for i in range(len(B))]
        # default_score = calculate_score_by_strategy(A, B, default_strategy)
        return default_strategy



#需要检查


import torch
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor


# 假设以下函数和类已在别处定义
# from your_module import (
#     TransformerMovePredictor,
#     strategy_TrueScore,
#     GA_Strategy,
#     calculate_future_score,
#     sample_order
# )

def _run_transformer_inference(TR_model, A_batch, B_batch, model_inference_indices, num_a, num_b, device):
    """
    私有辅助函数：专门执行 Transformer 模型的批量推理。
    返回一个字典，键是原始索引，值是Transformer生成的走法。
    """
    tr_moves_map = {}

    # 准备模型推理的批量数据
    all_features = []
    for idx in model_inference_indices:
        A = A_batch[idx]
        B = B_batch[idx]

        # --- 特征工程代码 (与原函数相同) ---
        B_counter = {}
        for val in B:
            B_counter[val] = B_counter.get(val, 0) + 1
        B_duplicates = sum(count - 1 for count in B_counter.values())

        intersection = set(A) & set(B)
        intersection_size = len(intersection)

        positions_in_A = {a_val: i for i, a_val in enumerate(A)}
        enhanced_sequence = []

        A_min, A_max = min(A), max(A)
        B_min, B_max = min(B), max(B)
        A_mean, A_std = np.mean(A), np.std(A)

        for i, val in enumerate(A):
            relative_position = i / num_a
            range_size = A_max - A_min
            min_max_scaled = (val - A_min) / range_size if range_size > 0 else 0.0
            is_extreme = 1.0 if (val == A_min or val == A_max) else 0.0
            z_score = (val - A_mean) / A_std if A_std > 0 else 0.0
            enhanced_sequence.append([
                val / sum(A), is_extreme, z_score, relative_position,
                min_max_scaled, intersection_size / num_a
            ])

        for i, val in enumerate(B):
            is_in_A = 1.0 if val in A else 0.0
            position_in_A = positions_in_A.get(val, -1)
            relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1
            is_extreme = 1.0 if (val == B_min or val == B_max) else 0.0
            remaining_B = B[i + 1:] if i < len(B) - 1 else []
            future_score = calculate_future_score(A, remaining_B)
            future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0
            enhanced_sequence.append([
                val / sum(B), future_score_ratio, is_in_A,
                relative_position_in_A, B_duplicates / num_b, is_extreme
            ])
        all_features.append(enhanced_sequence)

    # 批量推理
    input_tensor = torch.FloatTensor(np.array(all_features, dtype=np.float32)).to(device)
    with torch.no_grad():
        order_logits, pos_preds = TR_model(input_tensor)

    # 处理每个样本的结果
    for batch_idx, original_idx in enumerate(model_inference_indices):
        A = A_batch[original_idx]
        sample_order_logits = order_logits[batch_idx].unsqueeze(0)
        sample_pos_preds = pos_preds[batch_idx].unsqueeze(0)

        pred_order_indices = sample_order(sample_order_logits).squeeze(0).cpu().numpy()
        pred_moves_raw = sample_pos_preds.squeeze(0).cpu().numpy()

        pos_range_base = len(A)
        pred_moves_clipped = [int(np.clip(p, 0, pos_range_base + k)) for k, p in enumerate(pred_moves_raw)]

        moves_TR = [[int(pred_order_indices[k]), pred_moves_clipped[k]] for k in range(num_b)]
        tr_moves_map[original_idx] = moves_TR

    return tr_moves_map


def _run_ga_strategy(genomeforassist, A_batch, B_batch, model_inference_indices):
    """
    私有辅助函数：专门为需要推理的样本执行 GA 策略。
    返回一个字典，键是原始索引，值是GA生成的走法。
    """
    ga_moves_map = {}
    for idx in model_inference_indices:
        A = A_batch[idx]
        B = B_batch[idx]
        move_GA = GA_Strategy(genomeforassist, A, B)
        ga_moves_map[idx] = move_GA
    return ga_moves_map


def Transformer_predict_batch_plus_GA(A_batch, B_batch, genomeforassist, TR_model, num_a=6, num_b=3, device='cuda'):

    try:
        batch_size = len(A_batch)
        if not isinstance(TR_model, TransformerMovePredictor):
            raise ValueError(f"需要 TransformerMovePredictor 实例, 但得到 {type(TR_model)}")

        TR_model.eval()

        fast_path_indices = []
        model_inference_indices = []
        for i in range(batch_size):
            A, B = A_batch[i], B_batch[i]
            if len(A) != num_a or len(B) != num_b:
                raise ValueError(
                    f"样本 {i}: 期望输入 A 长度为 {num_a}, B 长度为 {num_b}, 但收到 A:{len(A)}, B:{len(B)}")

            if not set(A) & set(B) and len(B) == len(set(B)):
                fast_path_indices.append(i)
            else:
                model_inference_indices.append(i)

        moves_batch = [None] * batch_size

        # 1. 处理快速路径样本
        for idx in fast_path_indices:
            moves_batch[idx] = [[0, 0], [1, 0], [2, 0]]

        # 如果没有需要模型推理的样本，直接返回
        if not model_inference_indices:
            return moves_batch

        # 2. 并行处理需要模型推理的样本
        tr_moves_map = {}
        ga_moves_map = {}

        # 使用线程池并行执行 Transformer 推理和 GA 计算
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交Transformer任务
            tr_future = executor.submit(
                _run_transformer_inference,
                TR_model, A_batch, B_batch, model_inference_indices, num_a, num_b, device
            )
            # 提交GA任务
            ga_future = executor.submit(
                _run_ga_strategy,
                genomeforassist, A_batch, B_batch, model_inference_indices
            )
            # 获取结果
            tr_moves_map = tr_future.result()
            ga_moves_map = ga_future.result()

        # 3. 比较并选择最优结果
        for idx in model_inference_indices:
            A = A_batch[idx]
            B = B_batch[idx]

            moves_TR = tr_moves_map.get(idx)
            moves_GA = ga_moves_map.get(idx)

            # 如果某个策略没有返回结果（理论上不应发生），则设置一个极差的分数
            score_TR = strategy_TrueScore(A, B, moves_TR) if moves_TR else float('inf')
            score_GA = strategy_TrueScore(A, B, moves_GA) if moves_GA else float('inf')

            if score_GA < score_TR:
                moves_batch[idx] = moves_GA
            else:
                moves_batch[idx] = moves_TR

        return moves_batch

    except Exception as e:
        print(f"批量使用 Transformer+GA 预测时出错: {str(e)}")
        traceback.print_exc()
        print("返回默认策略")

        # 错误回退策略
        default_moves = []
        for i in range(batch_size):
            default_strategy = [[j, 1] for j in range(len(B_batch[i]))]
            default_moves.append(default_strategy)
        return default_moves

_json_cache = {}


def train():
    """
    加载训练数据并启动 Transformer(6x3) 训练过程
    """

    def load_jsonl_data(json_file_path):
        """
        加载JSONL数据并缓存
        """
        if json_file_path in _json_cache:
            return _json_cache[json_file_path]

        try:
            data = []
            with open(json_file_path, "r") as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"警告: 跳过无效的JSON行: {line[:50]}...")

            _json_cache[json_file_path] = data
            print(f"成功加载训练数据，样本数: {len(data)}")
            return data
        except FileNotFoundError:
            print(f"错误: 未找到 JSONL 文件 '{json_file_path}'")
            raise
        except Exception as e:
            print(f"加载数据时发生未知错误: {e}")
            raise

    try:
        # 使用缓存加载数据
        train_data = load_jsonl_data(jsonfile_path)
    except Exception as e:
        print(f"加载数据失败: {e}")
        exit(1)

    if not train_data:
        print("错误: 没有加载到有效的训练数据")
        exit(1)


    # 调用 train_model 时传递固定长度，并使用新的模型路径
    # 观察训练拟合后可以手动结束训练

    train_model(train_data, epochs=100, batch_size=2048, model_path=trans_path,
                num_a=6, num_b=3, warmup_epochs=5, lr_max=0.0005, lr_min=0.0000001,
                patience=2, min_delta=0.01)


_BLACK_HOLE_SINK = None


def black_hole(value):
    """
保证解释器不优化未使用的函数  以此确保时间开销统计的准确
    """
    global _BLACK_HOLE_SINK
    _BLACK_HOLE_SINK = value

_BLACK_HOLE_SINK = None



def black_hole(value):

    """

这里使用了一个小技巧,使用"black_hole"函数修改一个全局变量来保证解释器不会优化未使用的函数。 防止计算运行时间时不准确


    """

    global _BLACK_HOLE_SINK

    _BLACK_HOLE_SINK = value


def execution_time():

    total_time_transformer_P_GA = 0

    total_time_recursive = 0

    num_iterations = 20000  # 迭代次数


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transformer预测初始化

    num_a_test = 6

    num_b_test = 3


    d_model = 256

    nhead = 4

    num_encoder_layers = 3

    dim_feedforward = 512



    model1 = TransformerMovePredictor(

        num_a=num_a_test, num_b=num_b_test, d_model=d_model,

        nhead=nhead, num_encoder_layers=num_encoder_layers,

        dim_feedforward=dim_feedforward

    ).to(device)


    A_batch = []

    B_batch = []

    test_cases=[]


    # 辅助GA模型载入
    genomeforassist = load_best_genome("./trained/best_genome.pkl")

    ## Transformer模型载入
    model_path_1 = "./trained/transformer_move_predictor_6x3.pth"
    model1.load_state_dict(torch.load(model_path_1, map_location=device))

    # A,B数据准备
    for i in range(num_iterations):

        A, B = deal_cards_tool()

        A_batch.append(A)

        B_batch.append(B)


        test_cases.append((list(A), list(B)))


#计算Transform+GA 预测 运行的时间--------------------------------------------------------------

    start_time_tg = time.perf_counter()

    move_t = Transformer_predict_batch_plus_GA(A_batch, B_batch,genomeforassist, model1, num_a=num_a_test, num_b=num_b_test)


    # 因为得到的返回值move是批次的(列表),这里模拟将得到的列表中的元素取出操作消耗的时间

    for item in move_t:
        black_hole(item)


    end_time_tg = time.perf_counter()



    total_time_transformer_P_GA += (end_time_tg - start_time_tg)
#GA算法运行的时间 -----------------------------------------------------------------------------
    total_time_GA=0
    start_time_g = time.perf_counter()
    for A_copy, B_copy in test_cases:
        move_GA = GA_Strategy(genomeforassist, A_copy, A_copy)

        # 保证解释器不优化未使用的move_GA
        black_hole(move_GA)
    end_time_g = time.perf_counter()

    total_time_GA += (end_time_g - start_time_g)

#递归穷举运行的时间-----------------------------------------------------------------------------
    start_time_r = time.perf_counter()

    for A_copy, B_copy in test_cases:
        move_r = recursive_Strategy(A_copy, B_copy)

#保证解释器不优化未使用的move_r
        black_hole(move_r)

    end_time_r = time.perf_counter()

    total_time_recursive += (end_time_r - start_time_r)
#--------------------------------------------------------------------------------------------
    total_time_Transformer = 0
    start_time_t = time.perf_counter()
    for A_copy, B_copy in test_cases:
        move_T = Transformer_predict( A_copy, B_copy,model1)

        # 保证解释器不优化未使用的move_GA
        black_hole(move_T)
    end_time_t = time.perf_counter()

    total_time_Transformer += (end_time_t - start_time_t)





    print("\n--- Comparison Complete ---")

    print(f"Total time for Transformer+GA:      {total_time_transformer_P_GA:.6f} seconds")
    print(f"Total time for GA: {total_time_GA:.6f} seconds")
    print(f"Total time for Transformer:      {total_time_Transformer:.6f} seconds")
    print(f"Total time for recursive_Strategy: {total_time_recursive:.6f} seconds")





if __name__ == "__main__":

    # Todo 训练前注意路径!!!
    jsonfile_path = "json/data_Trans_fill.jsonl"
    trans_path="./trained/transformer_move_predictor_6x3.pth"

    # jsonfile_path = "json/data_Trans_skip.jsonl"
    # trans_path = "./trained/transformer_move_predictor_6x3_skip.pth"

    # train()

    execution_time()