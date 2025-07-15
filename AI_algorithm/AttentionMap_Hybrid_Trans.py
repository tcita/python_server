import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 步骤 1: 模型和数据准备代码 (与之前相同，无需修改)
# ==============================================================================

# --- 模型的基础模块 (从您的代码中复制) ---
class SelectivePositionalEncoding(nn.Module):
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
        x[:, :num_a, :] = x[:, :num_a, :] + self.pe[:, :num_a, :]
        return self.dropout(x)


# --- 您的数据准备函数 (简化版，用于预测) ---
def prepare_data_for_viz(A, B, num_a=6, num_b=3):
    if len(A) != num_a or len(B) != num_b: return None
    enhanced_sequence = [[val, 0, 0, 0, 0, 0] for val in A] + \
                        [[val, 0, 0, 0, 0, 0] for val in B]
    return np.array(enhanced_sequence, dtype=np.float32)


# ==============================================================================
# 步骤 2 & 3: 返回注意力的模型代码 (与之前相同，无需修改)
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


class HybridTransformerWithAttention(nn.Module):
    def __init__(self, input_dim=6, d_model=256, nhead=4, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, num_a=6, num_b=3):
        super().__init__()
        self.num_a = num_a
        self.num_b = num_b
        self.d_model = d_model
        seq_len = num_a + num_b
        self.input_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_encoder = SelectivePositionalEncoding(d_model, dropout, max_len=seq_len + 5)
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.order_head = nn.Linear(2 * d_model, num_b)
        self.pos_head = nn.Linear(2 * d_model, 1)

    def forward(self, src):
        if src.dim() == 2: src = src.unsqueeze(0)
        input_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        type_ids = torch.cat([torch.zeros(self.num_a, dtype=torch.long),
                              torch.ones(self.num_b, dtype=torch.long)], dim=0).to(src.device)
        type_ids = type_ids.unsqueeze(0).expand(src.size(0), -1)
        type_embedded = self.type_embed(type_ids)
        embedded = input_embedded + type_embedded
        embedded = self.pos_encoder(embedded, self.num_a)
        memory, attention_weights = self.transformer_encoder(embedded)
        b_features = memory[:, self.num_a:]
        b_global_feature = torch.mean(b_features, dim=1, keepdim=True)
        b_global_feature_expanded = b_global_feature.expand(-1, self.num_b, -1)
        combined_features = torch.cat([b_features, b_global_feature_expanded], dim=2)
        order_logits = self.order_head(combined_features)
        pos_preds = self.pos_head(combined_features).squeeze(-1)
        return order_logits, pos_preds, attention_weights


# ==============================================================================
# 步骤 4: 修正绘图函数和主分析函数
# ==============================================================================

def plot_attention_maps(attention_maps, input_A, input_B, num_heads):
    """
    【已修正】使用matplotlib和seaborn绘制平均后的注意力热力图。
    现在每层只绘制一个图。
    """
    num_layers = len(attention_maps)

    # attention_maps[i] 的形状是 (batch_size, seq_len, seq_len)

    # 创建标签
    labels = [f"A{i}({v})" for i, v in enumerate(input_A)] + \
             [f"B{i}({v})" for i, v in enumerate(input_B)]

    # --- 修正点 1: 更改子图布局为 1 行 x num_layers 列 ---
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5.5))
    if num_layers == 1:  # 确保axes在单层情况下也是一个列表
        axes = [axes]

    fig.suptitle("Transformer Attention Maps (Averaged over Heads)", fontsize=16)

    for layer_idx, layer_attention in enumerate(attention_maps):
        ax = axes[layer_idx]

        # --- 修正点 2: 直接从(batch_size, seq, seq)中提取二维矩阵 ---
        # 提取 batch 0 的注意力矩阵
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


def analyze_and_visualize_attention(model_path, sample_A, sample_B):
    """
    【已修正】主函数：加载模型，处理样本，并调用修正后的绘图函数。
    """
    print("--- 开始分析模型注意力权重 ---")

    model_params = {
        'input_dim': 6, 'd_model': 256, 'nhead': 4,
        'num_encoder_layers': 3, 'dim_feedforward': 512
    }

    model = HybridTransformerWithAttention(**model_params)

    try:
        # Pytorch 1.13+ 推荐使用 weights_only=True，更安全
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except TypeError:  # 兼容旧版Pytorch
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        model.load_state_dict(state_dict)
        model.eval()
        print(f"成功从 '{model_path}' 加载权重。")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return

    input_sequence = prepare_data_for_viz(sample_A, sample_B)
    if input_sequence is None:
        print("输入数据格式不正确。")
        return
    input_tensor = torch.FloatTensor(input_sequence)

    with torch.no_grad():
        _, _, attention_weights = model(input_tensor)

    print(f"成功获取了 {len(attention_weights)} 层的注意力权重。")
    print(f"每层的权重形状 (batch, seq, seq): {attention_weights[0].shape}")

    # --- 修正点 3: 将注意力头数 nhead 传递给绘图函数 ---
    plot_attention_maps(attention_weights, sample_A, sample_B, num_heads=model_params['nhead'])


# ==============================================================================
#  主程序入口
# ==============================================================================

if __name__ == "__main__":
    PTH_FILE_PATH = "./trained/Hybrid_Transformer.pth"

    A = [11, 13, 3, 10, 12, 6]
    B = [13, 8, 1]

    analyze_and_visualize_attention(
        model_path=PTH_FILE_PATH,
        sample_A=A,
        sample_B=B
    )