import sys
import os
import torch
from flask import Flask, request, jsonify
import pickle
from waitress import serve
import logging  # 1. 导入 logging 模块

# --- 日志配置 ---
# 2. 设置日志的基本配置
#    - level=logging.INFO: 表示 INFO, WARNING, ERROR, CRITICAL 级别的日志都会被显示
#    - format: 定义了每条日志的输出格式，包含 时间、日志级别、日志消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 假设 Trans.py 和 GA.py 在同一目录下或在 Python 的搜索路径中
from Trans import TransformerMovePredictor, Transformer_predict_batch_plus_GA
# from .Trans import TransformerMovePredictor, Transformer_predict_batch_plus_GA
# --- 全局变量和模型加载 ---

app = Flask(__name__)

# 定义设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和基因组变量
TR_MODEL = None
GENOME_FOR_ASSIST = None
NUM_A = 6
NUM_B = 3


def resource_path(relative_path):
    """
    获取资源的绝对路径，无论是从脚本运行还是从打包后的 EXE 运行。
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# 定义模型路径
MODEL_PATH = resource_path("trained/transformer_move_predictor_6x3.pth")
GENOME_PATH = resource_path("trained/best_genome.pkl")
# MODEL_PATH = resource_path("AI_algorithm/trained/transformer_move_predictor_6x3.pth")
# GENOME_PATH = resource_path("AI_algorithm/trained/best_genome.pkl")

def load_models():
    """
    加载 Transformer 模型和 GA 基因组到全局变量中。
    """
    global TR_MODEL, GENOME_FOR_ASSIST

    # 3. 将 print 语句改为 logging.info，让所有输出格式统一
    logging.info(f"服务器正在使用设备: {device}")
    logging.info(f"从以下路径加载模型: {MODEL_PATH}")
    logging.info(f"从以下路径加载基因组: {GENOME_PATH}")

    try:
        # ... (模型加载逻辑保持不变) ...
        d_model = 256
        nhead = 4
        num_encoder_layers = 3
        dim_feedforward = 512
        input_dim = 6

        TR_MODEL = TransformerMovePredictor(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=0.1, num_a=NUM_A, num_b=NUM_B
        ).to(device)

        TR_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        TR_MODEL.eval()
        logging.info(f"Transformer 模型从 '{MODEL_PATH}' 加载成功。")

        with open(GENOME_PATH, 'rb') as f:
            GENOME_FOR_ASSIST = pickle.load(f)
        logging.info(f"GA 基因组从 '{GENOME_PATH}' 加载成功。")

    except FileNotFoundError as e:
        logging.error(f"错误：模型或基因组文件未找到 - {e}")  # 改为 logging.error
        input("按 Enter 键退出...")
        exit(1)
    except Exception as e:
        logging.error(f"加载模型时发生未知错误: {e}", exc_info=True)  # exc_info=True 会自动附带 traceback 信息
        input("按 Enter 键退出...")
        exit(1)


# --- Flask API 路由 ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收客户端发送的 A 和 B 牌组，返回预测的出牌策略 (move)。
    """
    # 4. 记录每个收到的请求，这是调试的起点
    logging.info("================  收到新的预测请求  ================")

    if not request.is_json:
        logging.warning("请求格式错误: Content-Type 不是 application/json")
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    # 5. 详细记录收到的数据内容，这是最关键的调试信息
    logging.info(f"收到的原始数据: {data}")

    A = data.get('A')
    B = data.get('B')

    # ... (输入验证逻辑保持不变) ...
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
        A_batch = [A]
        B_batch = [B]

        logging.info(f"准备调用AI模型进行预测，输入牌组 A: {A_batch}, B: {B_batch}")

        # --- 调用核心预测函数 ---
        moves_batch = Transformer_predict_batch_plus_GA(
            A_batch=A_batch, B_batch=B_batch, genomeforassist=GENOME_FOR_ASSIST,
            TR_model=TR_MODEL, num_a=NUM_A, num_b=NUM_B, device=device
        )

        predicted_move = moves_batch[0]
        # 6. 记录AI模型的输出结果
        logging.info(f"预测成功！AI推荐出牌: {predicted_move}")

        # 返回预测结果
        return jsonify({
            "success": True,
            "predicted_move": predicted_move
        })

    except Exception as e:
        # 7. 在发生未知错误时，记录详细的错误信息
        logging.error(f"预测过程中发生严重错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误，无法完成预测"}), 500


# --- 主程序入口 ---
if __name__ == '__main__':
    # 在启动服务器前先加载模型
    load_models()

    logging.info("模型加载完毕，准备启动服务器...")
    logging.info("服务器将在 http://0.0.0.0:5000 上监听请求")

    serve(app, host='0.0.0.0', port=5000)