import torch
from flask import Flask, request, jsonify
import numpy as np
import pickle

# 假设 Trans.py 和 GA.py 在同一目录下或在 Python 的搜索路径中
# 从 Trans.py 导入所需的类和函数
from Trans import TransformerMovePredictor, Transformer_predict_batch_plus_GA
# 从 GA.py 导入所需的函数 (尽管在 Transformer_predict_batch_plus_GA 中调用，但为了清晰起见)
from GA import GA_Strategy

# --- 全局变量和模型加载 ---

app = Flask(__name__)

# 定义设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"服务器正在使用设备: {device}")

# 初始化模型和基因组变量
TR_MODEL = None
GENOME_FOR_ASSIST = None
MODEL_PATH = "./trained/transformer_move_predictor_6x3.pth"
GENOME_PATH = "./trained/best_genome.pkl"
NUM_A = 6
NUM_B = 3


def load_models():
    """
    加载 Transformer 模型和 GA 基因组到全局变量中。
    """
    global TR_MODEL, GENOME_FOR_ASSIST

    try:
        # --- 加载 Transformer 模型 ---
        # 模型参数 (需要与训练时一致)
        d_model = 256
        nhead = 4
        num_encoder_layers = 3
        dim_feedforward = 512
        input_dim = 6  # 特征维度

        # 实例化模型
        TR_MODEL = TransformerMovePredictor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_a=NUM_A,
            num_b=NUM_B
        ).to(device)

        # 加载模型状态字典
        TR_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        TR_MODEL.eval()  # 设置为评估模式
        print(f"Transformer 模型从 '{MODEL_PATH}' 加载成功。")

        # --- 加载 GA 基因组 ---
        with open(GENOME_PATH, 'rb') as f:
            GENOME_FOR_ASSIST = pickle.load(f)
        print(f"GA 基因组从 '{GENOME_PATH}' 加载成功。")

    except FileNotFoundError as e:
        print(f"错误：模型或基因组文件未找到 - {e}")
        # 在这种关键错误下，服务器无法正常工作，可以选择退出
        exit(1)
    except Exception as e:
        print(f"加载模型时发生未知错误: {e}")
        exit(1)


# --- Flask API 路由 ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收客户端发送的 A 和 B 牌组，返回预测的出牌策略 (move)。
    """
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    A = data.get('A')
    B = data.get('B')

    # --- 输入验证 ---
    if not A or not B:
        return jsonify({"error": "请求体中必须包含 'A' 和 'B' 字段"}), 400

    if not isinstance(A, list) or not isinstance(B, list):
        return jsonify({"error": "'A' 和 'B' 必须是列表"}), 400

    if len(A) != NUM_A or len(B) != NUM_B:
        return jsonify({
            "error": f"输入牌的数量不正确。期望 A 有 {NUM_A} 张, B 有 {NUM_B} 张。",
            "received": f"A: {len(A)}, B: {len(B)}"
        }), 400

    try:
        # 将单个输入包装成批次 (batch)
        A_batch = [A]
        B_batch = [B]

        # --- 调用核心预测函数 ---
        # 注意：Transformer_predict_batch_plus_GA 的原始实现返回了两个值，
        # 但在您的代码中，第二个返回值被注释掉了，所以我们只关心第一个返回值。
        # 如果您的函数确实返回两个值，可以写成：moves_batch, scores_batch = ...
        moves_batch = Transformer_predict_batch_plus_GA(
            A_batch=A_batch,
            B_batch=B_batch,
            genomeforassist=GENOME_FOR_ASSIST,
            TR_model=TR_MODEL,
            num_a=NUM_A,
            num_b=NUM_B,
            device=device
        )

        # 从批次结果中提取第一个（也是唯一一个）预测结果
        predicted_move = moves_batch[0]

        # 返回预测结果
        return jsonify({
            "success": True,
            "predicted_move": predicted_move
        })

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        # 在生产环境中，您可能希望记录更详细的错误信息
        import traceback
        traceback.print_exc()
        return jsonify({"error": "服务器内部错误，无法完成预测"}), 500


# --- 主程序入口 ---
if __name__ == '__main__':
    # 在启动服务器前先加载模型
    load_models()

    # 启动 Flask 开发服务器
    # 在生产环境中，应使用 Gunicorn 或 uWSGI 等 WSGI 服务器
    app.run(host='0.0.0.0', port=5000, debug=True)