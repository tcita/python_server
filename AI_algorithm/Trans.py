import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import signal
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict

from AI_algorithm.tool.tool import calculate_score_by_strategy, simulate_insertion_tool, calculate_future_score


# 使用LRU缓存替代无限制缓存
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        # 将访问的项移到队尾（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # 已存在则更新
            self.cache.move_to_end(key)
        self.cache[key] = value
        # 检查容量并移除最久未使用的项
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# 创建有限容量的缓存
_preprocess_cache = LRUCache(capacity=100000)  # 调整为合适的缓存大小
_prediction_cache = LRUCache(capacity=50000)
_json_cache = {}  # 这个缓存可以保持原样，因为通常只有一个文件

jsonfilename = "json/normal_scores_uniq.json"  # 请确保你的 JSON 文件路径正确

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
    def __init__(self, d_model, dropout=0.1, max_len=20):  # 增加 max_len 以确保安全
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
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):  # 修改input_dim为6
        super(TransformerMovePredictor, self).__init__()
        # 验证传入的 num_a 和 num_b 是否符合预期（可选但推荐）
        if num_a != 6 or num_b != 3:
            print(f"警告: TransformerMovePredictor 初始化期望 num_a=6, num_b=3, 但收到了 num_a={num_a}, num_b={num_b}")

        self.num_a = num_a
        self.num_b = num_b
        self.seq_len = num_a + num_b  # 现在是 6 + 3 = 9
        self.d_model = d_model

        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)  # 0 for A, 1 for B
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
            src = src.unsqueeze(-1)  # (batch, seq_len, 1)

        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)

        # type_ids: num_a 个 0, num_b 个 1
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)  # (batch_size, seq_len)
        type_embedded = self.type_embed(type_ids)

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


# 优化的数据预处理函数
def prepare_data_transformer(sample: dict, num_a=6, num_b=3):
    """
    优化的数据预处理：构建输入序列和目标输出，添加修改后的特征工程。
    使用缓存减少重复计算。
    """
    # 使用样本的唯一标识作为缓存键
    cache_key = (tuple(sample["A"]), tuple(sample["B"]), tuple(tuple(move) for move in sample["best_moves"]))

    # 检查缓存中是否已存在处理结果
    cached_result = _preprocess_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    A = sample["A"]
    B = sample["B"]
    best_moves = sample["best_moves"]

    if len(A) != num_a or len(B) != num_b:
        return None, None, None

    # 使用预先计算来减少冗余计算
    A_sum = sum(A)
    B_sum = sum(B)

    # 更高效地计算交集
    A_set = set(A)
    B_set = set(B)
    intersection = A_set & B_set
    intersection_size = len(intersection)
    intersection_ratio = intersection_size / num_b

    # 计算B序列中重复的元素
    B_counter = {}
    for val in B:
        B_counter[val] = B_counter.get(val, 0) + 1
    B_duplicates = sum(count - 1 for count in B_counter.values())

    # 预计算位置信息
    positions_in_A = {a_val: i for i, a_val in enumerate(A)}

    # 创建增强的输入序列
    enhanced_sequence = np.zeros((num_a + num_b, 6), dtype=np.float32)  # 预分配数组

    # 处理A序列
    for i, val in enumerate(A):
        is_in_intersection = 1.0 if val in intersection else 0.0
        relative_position = i / num_a
        relative_size = val / A_sum if A_sum != 0 else 0

        enhanced_sequence[i] = [
            val,
            val / A_sum if A_sum != 0 else 0,
            is_in_intersection,
            relative_position,
            relative_size,
            intersection_ratio
        ]

    # 处理B序列
    for i, val in enumerate(B):
        is_in_A = 1.0 if val in A_set else 0.0
        position_in_A = positions_in_A.get(val, -1)
        relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1

        # 计算未来得分特征
        remaining_B = B[i + 1:] if i < len(B) - 1 else []
        future_score = calculate_future_score(A, remaining_B)
        future_score_ratio = future_score / (A_sum + B_sum) if (A_sum + B_sum) > 0 else 0

        enhanced_sequence[num_a + i] = [
            val,
            future_score_ratio,
            is_in_A,
            relative_position_in_A,
            B_duplicates / num_b,
            intersection_ratio
        ]

    # 转换目标数据
    order_target = np.array([move[0] for move in best_moves], dtype=np.int64)
    pos_target = np.array([move[1] for move in best_moves], dtype=np.float32)

    if len(set(order_target)) != num_b:
        return None, None, None

    # 将结果存入缓存
    result = (enhanced_sequence, order_target, pos_target)
    _preprocess_cache.put(cache_key, result)

    return result


# 定义全局变量
stop_training = False


def signal_handler(sig, frame):
    global stop_training
    print("\n检测到手动停止信号，准备保存当前模型...")
    stop_training = True


# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)


def lr_warmup(epoch, lr_max, warmup_epochs):
    if epoch < warmup_epochs:
        warmup_lr = lr_max * (epoch + 1) / warmup_epochs
        return warmup_lr
    else:
        return lr_max


# 优化的数据加载器
class LazyDataLoader:
    """惰性数据加载器，只在需要时加载和处理数据"""

    def __init__(self, data, batch_size, num_a=6, num_b=3, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.num_a = num_a
        self.num_b = num_b
        self.shuffle = shuffle
        self.indices = list(range(len(data)))
        self.cursor = 0

        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.cursor >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size

        inputs, order_targets, pos_targets = [], [], []

        for idx in batch_indices:
            sample = self.data[idx]
            input_seq, order_tgt, pos_tgt = prepare_data_transformer(sample, num_a=self.num_a, num_b=self.num_b)
            if input_seq is None:
                continue
            inputs.append(input_seq)
            order_targets.append(order_tgt)
            pos_targets.append(pos_tgt)

        if not inputs:
            # 如果这个批次没有有效样本，尝试获取下一个批次
            return self.__next__()

        inputs = torch.FloatTensor(np.array(inputs)).to(device)
        order_targets = torch.LongTensor(np.array(order_targets)).to(device)
        pos_targets = torch.FloatTensor(np.array(pos_targets)).to(device)

        return inputs, order_targets, pos_targets

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


# 优化的训练函数
def train_model(train_data, epochs=300, batch_size=64, model_path="./trained/transformer_move_predictor_6x3.pth",
                num_a=6, num_b=3, warmup_epochs=10, lr_max=0.0001, lr_min=0.0000005,
                patience=10, min_delta=0.01):
    """
    优化的训练函数：使用惰性数据加载，更有效的内存管理和批处理
    """
    global stop_training

    d_model = 256
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # 初始化模型
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

    # 学习率调度
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min)

    # 早停相关变量
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 创建惰性数据加载器
    data_loader = LazyDataLoader(train_data, batch_size, num_a=num_a, num_b=num_b, shuffle=True)

    try:
        for epoch in range(epochs):
            if stop_training:
                break

            total_loss = 0.0
            batch_count = 0

            # 使用惰性数据加载器
            for inputs, order_targets, pos_targets in data_loader:
                if stop_training:
                    break

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # 主动释放不需要的张量
                del inputs, order_targets, pos_targets, order_logits, pos_preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 在每个epoch后更新余弦退火学习率
            if epoch >= warmup_epochs:
                scheduler_cosine.step()

            avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {new_lr:.6f}")

            # 早停检查
            if avg_loss < best_loss - min_delta:  # 有显著改善
                best_loss = avg_loss
                patience_counter = 0
                # 保存当前最佳模型状态
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"发现新的最佳模型，损失值: {best_loss:.4f}")
            else:  # 没有显著改善
                patience_counter += 1
                print(f"未发现显著改善，耐心计数: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"早停触发！连续 {patience} 个epoch无显著改善")
                    # 恢复到最佳模型状态
                    if best_model_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
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
                model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                print("已加载训练过程中的最佳模型状态")

            torch.save(model.state_dict(), model_path)
            print(f"Transformer 模型已保存至 {model_path}")
        except Exception as save_error:
            print(f"保存模型时发生错误: {str(save_error)}")

    return model


def Transformer_predict(A, B, model, num_a=6, num_b=3):
    """
    使用训练好的 Transformer 模型进行预测 (针对 A=6, B=3)。
    """
    try:
        # 缓存键
        cache_key = (tuple(A), tuple(B), id(model))

        # 检查缓存
        cached_result = _prediction_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

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

        # 应用相同的特征工程但更高效
        A_sum = sum(A)
        B_sum = sum(B)

        A_set = set(A)
        B_set = set(B)
        intersection = A_set & B_set
        intersection_size = len(intersection)
        intersection_ratio = intersection_size / num_b

        # 使用字典推导而不是循环来提高效率
        B_counter = {}
        for val in B:
            B_counter[val] = B_counter.get(val, 0) + 1
        B_duplicates = sum(count - 1 for count in B_counter.values())

        positions_in_A = {a_val: i for i, a_val in enumerate(A)}

        # 预分配数组
        enhanced_sequence = np.zeros((num_a + num_b, 6), dtype=np.float32)

        # 处理A序列
        for i, val in enumerate(A):
            is_in_intersection = 1.0 if val in intersection else 0.0
            relative_position = i / num_a
            relative_size = val / A_sum if A_sum != 0 else 0

            enhanced_sequence[i] = [
                val,
                val / A_sum if A_sum != 0 else 0,
                is_in_intersection,
                relative_position,
                relative_size,
                intersection_ratio
            ]

        # 处理B序列
        for i, val in enumerate(B):
            is_in_A = 1.0 if val in A_set else 0.0
            position_in_A = positions_in_A.get(val, -1)
            relative_position_in_A = position_in_A / num_a if position_in_A >= 0 else -0.1

            remaining_B = B[i + 1:] if i < len(B) - 1 else []
            future_score = calculate_future_score(A, remaining_B)
            future_score_ratio = future_score / (A_sum + B_sum) if (A_sum + B_sum) > 0 else 0

            enhanced_sequence[num_a + i] = [
                val,
                future_score_ratio,
                is_in_A,
                relative_position_in_A,
                B_duplicates / num_b,
                intersection_ratio
            ]

        input_tensor = torch.FloatTensor(enhanced_sequence).unsqueeze(0).to(device)

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

        result = (best_moves, pred_score)
        _prediction_cache.put(cache_key, result)

        # 释放不再需要的张量
        del input_tensor, order_logits, pos_preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except Exception as e:
        print(f"使用 Transformer 预测时出错 (6x3): A={A}, B={B}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("返回默认策略")
        default_strategy = [[i, 1] for i in range(len(B))]
        default_score = calculate_score_by_strategy(A, B, default_strategy)
        return default_strategy, default_score


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
            print(f"正在加载JSON数据文件: {json_file_path}")
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

    # 调用 train_model 时传递固定长度，并使用新的模型路径
    # epochs=1000  warmup_epochs=50 是对应10万个样本的
    train_model(train_data, epochs=1000, batch_size=128, model_path="./trained/transformer_move_predictor_6x3.pth",
                num_a=6, num_b=3, warmup_epochs=10, lr_max=0.00001, lr_min=0.0000001,
                patience=20, min_delta=0.01)


if __name__ == "__main__":
    train()