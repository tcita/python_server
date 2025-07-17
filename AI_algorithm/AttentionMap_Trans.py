import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm


# ==============================================================================
# 步骤 0: 依赖模块 (与之前相同)
# ==============================================================================

def calculate_future_score_default(A, B):
    return sum(A) - sum(B) if A and B else (sum(A) if A else 0)


def prepare_data_for_viz(sample: dict, num_a=6, num_b=3):
    A = sample["A"]
    B = sample["B"]
    if len(A) != num_a or len(B) != num_b: return None
    B_counter = {val: B.count(val) for val in B}
    B_duplicates = sum(count - 1 for count in B_counter.values())
    intersection = set(A) & set(B)
    intersection_size = len(intersection)
    positions_in_A = {a_val: i for i, a_val in enumerate(A)}
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
        remaining_B = B[i + 1:] if i < len(B) - 1 else []
        future_score = calculate_future_score_default(A, remaining_B)
        future_score_ratio = future_score / (sum(A) + sum(B)) if (sum(A) + sum(B)) > 0 else 0
        enhanced_sequence.append(
            [val / sum(B), future_score_ratio, is_in_A, relative_position_in_A, B_duplicates / num_b, is_extreme])
    return np.array(enhanced_sequence, dtype=np.float32)


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


# ==============================================================================
# 步骤 1: 自定义 Transformer 模块 (与之前相同)
# ==============================================================================
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        x = src
        attn_output, attn_weights = self.self_attn(
            x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=True
        )
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_output))
        return x, attn_weights


class CustomTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
        output = src
        attention_weights_list = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights_list.append(attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, attention_weights_list


# ==============================================================================
# 步骤 2: 两个模型的 "WithAttention" 版本定义
# ==============================================================================

# --- 模型一: SetProcessorTransformerWithAttention (两阶段) ---
class SetProcessorTransformerWithAttention(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=4, num_b_encoder_layers=2,
                 num_main_encoder_layers=3, dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(SetProcessorTransformerWithAttention, self).__init__()
        self.num_a, self.num_b, self.d_model = num_a, num_b, d_model
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_a + 5)
        b_encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.b_set_processor = CustomTransformerEncoder(b_encoder_layer, num_b_encoder_layers)
        main_encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.main_encoder = CustomTransformerEncoder(main_encoder_layer, num_main_encoder_layers)
        self.slot_queries = nn.Parameter(torch.randn(1, num_b, d_model))
        self.pos_head = nn.Linear(d_model, 1)

    def forward(self, src):
        if src.dim() == 2: src = src.unsqueeze(0)
        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long), torch.ones(self.num_b, dtype=torch.long)],
                             dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)
        embedded = input_embedded + type_embedded
        embedded_a, embedded_b = embedded[:, :self.num_a], embedded[:, self.num_a:]
        contextual_b_features, _ = self.b_set_processor(embedded_b)  # 我们忽略 b_set_processor 的注意力
        encoded_a = self.pos_encoder(embedded_a)
        combined_sequence = torch.cat([encoded_a, contextual_b_features], dim=1)
        _, main_encoder_attn_weights = self.main_encoder(combined_sequence)  # 只关心 main_encoder 的注意力
        return main_encoder_attn_weights


# --- 模型二: TransformerMovePredictorWithAttention (单阶段) ---
class TransformerMovePredictorWithAttention(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=4, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super(TransformerMovePredictorWithAttention, self).__init__()
        self.num_a, self.num_b, self.d_model = num_a, num_b, d_model
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_a + num_b + 5)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_encoder_layers)

        # 【修正】将原始模型中的预测头加回来，以匹配权重文件
        self.order_head = nn.Linear(num_b * d_model, num_b * num_b)
        self.pos_head = nn.Linear(num_b * d_model, num_b)

    def forward(self, src):
        if src.dim() == 2: src = src.unsqueeze(0)
        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long), torch.ones(self.num_b, dtype=torch.long)],
                             dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)
        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded)

        # 注意，我们仍然只返回注意力权重
        memory, attention_weights = self.transformer_encoder(embedded)

        # (我们不需要计算最终的logits，所以这里用memory而不是b_features)

        return attention_weights
# ==============================================================================
# 步骤 3: 【全新】对比绘图和对比分析函数
# ==============================================================================

def plot_comparison_attention_maps(title, attn_maps_model1, attn_maps_model2, model1_name, model2_name, labels,
                                   num_heads):
    """一个专门用于上下对比两个模型注意力图的绘图函数。"""
    num_layers = max(len(attn_maps_model1), len(attn_maps_model2))
    fig, axes = plt.subplots(2, num_layers, figsize=(6 * num_layers + 1, 11), squeeze=False)
    fig.suptitle(title, fontsize=20, y=0.98)

    model_maps = [attn_maps_model1, attn_maps_model2]
    model_names = [model1_name, model2_name]

    for row in range(2):
        for col in range(num_layers):
            ax = axes[row, col]

            if col < len(model_maps[row]):
                attention_matrix = model_maps[row][col][0].detach().cpu().numpy()
                sns.heatmap(attention_matrix, ax=ax, cmap="viridis", cbar=True,
                            xticklabels=labels, yticklabels=labels, vmin=0, vmax=0.2)  # 使用 vmin/vmax 统一色标
                ax.set_title(f"Layer {col + 1}")
            else:
                ax.axis('off')  # 如果某个模型层数较少，则隐藏多余的子图

            if col == 0:
                ax.set_ylabel(model_names[row], fontsize=14, weight='bold')

    # 【修正】替换 tight_layout 为 subplots_adjust 以精确控制间距
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(
        hspace=0.4,  # hspace 控制子图之间的垂直间距
        wspace=0.3  # wspace 控制子图之间的水平间距
    )

    plt.show()

def analyze_and_compare_models(model1_path, model2_path, data_samples, num_samples_to_use=500):
    """主函数：加载两个模型，计算它们可比部分的平均注意力，并并排绘图。"""
    print("--- 开始对比两个模型的全局平均注意力 ---")

    # --- 模型参数定义 ---
    # 确保这里的参数与您训练的两个模型一致
    model1_params = {'input_dim': 6, 'd_model': 256, 'nhead': 4, 'num_b_encoder_layers': 2,
                     'num_main_encoder_layers': 3, 'dim_feedforward': 512}
    model2_params = {'input_dim': 6, 'd_model': 256, 'nhead': 4, 'num_encoder_layers': 3, 'dim_feedforward': 512}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载两个模型 ---
    model1 = SetProcessorTransformerWithAttention(**model1_params).to(device)
    model2 = TransformerMovePredictorWithAttention(**model2_params).to(device)

    try:
        model1.load_state_dict(torch.load(model1_path, map_location=device))
        model1.eval()
        print(f"成功加载模型一: {model1_path}")
        model2.load_state_dict(torch.load(model2_path, map_location=device))
        model2.eval()
        print(f"成功加载模型二: {model2_path}")
    except Exception as e:
        print(f"加载模型失败: {e}");
        return

    # --- 初始化注意力累加器 ---
    num_layers1 = model1_params['num_main_encoder_layers']
    num_layers2 = model2_params['num_encoder_layers']
    attn_sum1 = [torch.zeros(1, 9, 9, device=device) for _ in range(num_layers1)]
    attn_sum2 = [torch.zeros(1, 9, 9, device=device) for _ in range(num_layers2)]
    processed_count = 0

    # --- 循环处理样本，为两个模型同时累加注意力 ---
    samples_to_process = data_samples[:min(num_samples_to_use, len(data_samples))]
    with torch.no_grad():
        for sample in tqdm(samples_to_process, desc="Comparing Models"):
            input_sequence = prepare_data_for_viz(sample)
            if input_sequence is None: continue
            input_tensor = torch.FloatTensor(input_sequence).to(device)

            # 模型一 (只获取 main_encoder 的权重)
            attn1 = model1(input_tensor)
            for i in range(num_layers1):
                attn_sum1[i] += attn1[i].detach()

            # 模型二 (获取其唯一编码器的权重)
            attn2 = model2(input_tensor)
            for i in range(num_layers2):
                attn_sum2[i] += attn2[i].detach()

            processed_count += 1

    if processed_count == 0: print("没有有效的样本被处理。"); return

    # --- 计算平均值并绘图 ---
    avg_attn1 = [s / processed_count for s in attn_sum1]
    avg_attn2 = [s / processed_count for s in attn_sum2]

    labels = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(3)]
    plot_comparison_attention_maps(
        f"Global Attention Comparison (over {processed_count} samples)",
        avg_attn1, avg_attn2,
        "Model 1 (2-Stage)",
        "Model 2 (1-Stage)",
        labels, model1_params['nhead']
    )


def load_jsonl_data(json_file_path):
    try:
        with open(json_file_path, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(f"成功加载对比数据: {len(data)} 条")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}");
        return None


# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == "__main__":
    # --- 【请修改】这里填写您的两个模型的路径 ---
    MODEL1_PATH = "./trained/SetProcessor_Transformer.pth"
    MODEL2_PATH = "./trained/transformer_move_predictor_6x3.pth"

    # --- 【请修改】这里填写用于对比分析的数据文件路径 ---
    DATA_PATH = "json/data_Attention_Map.jsonl"

    # 加载数据
    all_data = load_jsonl_data(DATA_PATH)

    if all_data:
        analyze_and_compare_models(
            model1_path=MODEL1_PATH,
            model2_path=MODEL2_PATH,
            data_samples=all_data,
            num_samples_to_use=5000  # 可调整用于平均的样本数
        )