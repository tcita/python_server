import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 步骤 1: 复制新模型所需的依赖模块和数据准备函数
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    从您的新模型代码中复制的 PositionalEncoding 类。
    """

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


def prepare_data_for_viz_v2(A, B, num_a=6, num_b=3, input_dim=6):
    """
    一个简化的数据准备函数，仅用于生成正确的输入形状以进行可视化。
    """
    if len(A) != num_a or len(B) != num_b: return None
    # 创建一个具有正确形状 (9, 6) 的占位符序列
    placeholder_sequence = np.zeros((num_a + num_b, input_dim), dtype=np.float32)
    return placeholder_sequence


# ==============================================================================
# 步骤 2: 创建可以返回注意力权重的自定义Transformer模块
# (这部分与之前的代码完全相同，可以直接复用)
# ==============================================================================

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x, attn_weights = self.self_attn(src, src, src,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=True)
        x = self.dropout1(x)
        x = self.norm1(src + x)
        x = self.norm2(x + self._ff_block(x))
        return x, attn_weights


class CustomTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attention_weights_list = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights_list.append(attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, attention_weights_list


# ==============================================================================
# 步骤 3: 专门为 TransformerMovePredictor 定义一个带注意力输出的新模型
# ==============================================================================

class TransformerMovePredictorWithAttention(nn.Module):
    """
    这个模型的结构与您原始的 TransformerMovePredictor 完全相同，
    因此可以直接加载您训练好的 .pth 文件。
    唯一的区别是它使用自定义的Encoder来捕获注意力权重。
    """

    def __init__(self, input_dim=6, d_model=256, nhead=4, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super().__init__()
        self.num_a = num_a
        self.num_b = num_b
        self.seq_len = num_a + num_b
        self.d_model = d_model

        # --- 所有层的名称和参数必须与原始模型完全一致 ---
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.seq_len + 5)

        # --- 使用我们的自定义模块 ---
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # -------------------------

        # --- 输出头也必须完全一致 ---
        self.order_head = nn.Linear(num_b * d_model, num_b * num_b)
        self.pos_head = nn.Linear(num_b * d_model, num_b)

    def forward(self, src):
        # --- 前半部分与原始模型完全相同 ---
        if src.dim() == 2:
            src = src.unsqueeze(0)  # 修正：原始模型假设输入是 (batch, seq, dim)，所以这里应该是 unsqueeze(0)

        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)
        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded)

        # --- 获取注意力权重 ---
        memory, attention_weights = self.transformer_encoder(embedded)
        # --------------------

        # --- 后半部分与原始模型完全相同 ---
        b_features = memory[:, self.num_a:]
        b_features_flat = b_features.reshape(b_features.size(0), -1)
        order_logits_flat = self.order_head(b_features_flat)
        order_logits = order_logits_flat.view(-1, self.num_b, self.num_b)
        pos_preds = self.pos_head(b_features_flat)

        # --- 返回额外的信息 ---
        return order_logits, pos_preds, attention_weights


# ==============================================================================
# 步骤 4: 绘图和主分析函数 (与之前的代码基本相同)
# ==============================================================================

def plot_attention_maps(attention_maps, input_A, input_B, num_heads):
    """
    绘制平均后的注意力热力图。
    """
    num_layers = len(attention_maps)
    labels = [f"A{i}({v})" for i, v in enumerate(input_A)] + \
             [f"B{i}({v})" for i, v in enumerate(input_B)]

    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5.5))
    if num_layers == 1: axes = [axes]

    fig.suptitle("Transformer Attention Maps (Averaged over Heads)", fontsize=16)

    for layer_idx, layer_attention in enumerate(attention_maps):
        ax = axes[layer_idx]
        attention_matrix = layer_attention[0].detach().cpu().numpy()

        sns.heatmap(attention_matrix, ax=ax, cmap="viridis", cbar=True)

        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=90, ha="center")
        ax.set_yticklabels(labels, rotation=0, va="center")

        ax.set_title(f"Layer {layer_idx + 1} (Avg. of {num_heads} Heads)")
        ax.set_xlabel("Key (Attending From)")

    axes[0].set_ylabel("Query (Attending To)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def analyze_and_visualize_attention_v2(model_path, sample_A, sample_B):
    """
    主函数：加载新模型，处理样本，并调用绘图函数。
    """
    print("--- 开始分析新 Transformer 模型注意力权重 ---")

    # --- 1. 定义模型超参数 (必须与 train_model 函数中的一致) ---
    model_params = {
        'input_dim': 6, 'd_model': 256, 'nhead': 4,
        'num_encoder_layers': 3, 'dim_feedforward': 512,
        'num_a': 6, 'num_b': 3
    }

    # --- 2. 实例化我们带注意力输出的新模型 ---
    model = TransformerMovePredictorWithAttention(**model_params)

    # --- 3. 加载您训练好的权重 ---
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print(f"成功从 '{model_path}' 加载权重。")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return

    # --- 4. 准备输入数据 ---
    input_sequence = prepare_data_for_viz_v2(sample_A, sample_B)
    if input_sequence is None:
        print("输入数据格式不正确。")
        return
    input_tensor = torch.FloatTensor(input_sequence)

    # --- 5. 获取模型输出和注意力权重 ---
    with torch.no_grad():
        _, _, attention_weights = model(input_tensor)

    print(f"成功获取了 {len(attention_weights)} 层的注意力权重。")
    print(f"每层的权重形状 (batch, seq, seq): {attention_weights[0].shape}")

    # --- 6. 调用绘图函数 ---
    plot_attention_maps(attention_weights, sample_A, sample_B, num_heads=model_params['nhead'])


# ==============================================================================
#  主程序入口: 在这里定义您想分析的样本
# ==============================================================================

if __name__ == "__main__":
    # 您的新模型 .pth 文件路径
    # 根据您的代码，它应该位于 "./trained/transformer_move_predictor_6x3.pth"
    PTH_FILE_PATH_V2 = "./trained/transformer_move_predictor_6x3.pth"

    # 定义一个您想要分析的具体样本
    A = [11, 13, 3, 10, 12, 6]
    B = [13, 8, 1]

    # 运行新模型的分析和可视化
    analyze_and_visualize_attention_v2(
        model_path=PTH_FILE_PATH_V2,
        sample_A=A,
        sample_B=B
    )