import sys
import os
import torch
from flask import Flask, request, jsonify
import pickle
from waitress import serve
import logging  # 1. 导入 logging 模块 (Import logging module)

# --- 日志配置 (Logging Configuration) ---
# 2. 设置日志的基本配置 (Set up basic logging configuration)
#    - level=logging.INFO: 表示 INFO, WARNING, ERROR, CRITICAL 级别的日志都会被显示
#    - format: 定义了每条日志的输出格式，包含 时间、日志级别、日志消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 假设 Trans.py 和 GA.py 在同一目录下或在 Python 的搜索路径中
# from Trans import TransformerMovePredictor, Transformer_predict_batch_plus_GA
from .Trans import TransformerMovePredictor, Transformer_predict_batch_plus_GA
# --- 全局变量和模型加载 (Global Variables and Model Loading) ---

app = Flask(__name__)

# 定义设备 (GPU 或 CPU) (Define device - GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和基因组变量 (Initialize model and genome variables)
TR_MODEL = None
GENOME_FOR_ASSIST = None
NUM_A = 6
NUM_B = 3


def resource_path(relative_path):
    """
    获取资源的绝对路径，无论是从脚本运行还是从打包后的 EXE 运行。
    (Get absolute path of resource, whether running from script or packaged EXE)
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# 定义模型路径 (Define model paths)
MODEL_PATH = resource_path("trained/transformer_move_predictor_6x3.pth")
GENOME_PATH = resource_path("trained/best_genome.pkl")
# MODEL_PATH = resource_path("AI_algorithm/trained/transformer_move_predictor_6x3.pth")
# GENOME_PATH = resource_path("AI_algorithm/trained/best_genome.pkl")

def load_models():
    """
    加载 Transformer 模型和 GA 基因组到全局变量中。
    (Load Transformer model and GA genome into global variables)
    """
    global TR_MODEL, GENOME_FOR_ASSIST

    # 3. 将 print 语句改为 logging.info，让所有输出格式统一 (Use logging.info for consistent output format)
    logging.info(f"Server is using device: {device}")
    logging.info(f"Loading model from path: {MODEL_PATH}")
    logging.info(f"Loading genome from path: {GENOME_PATH}")

    try:
        # ... (模型加载逻辑保持不变) (Model loading logic remains unchanged) ...
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
        logging.info(f"Transformer model successfully loaded from '{MODEL_PATH}'")

        with open(GENOME_PATH, 'rb') as f:
            GENOME_FOR_ASSIST = pickle.load(f)
        logging.info(f"GA genome successfully loaded from '{GENOME_PATH}'")

    except FileNotFoundError as e:
        logging.error(f"Error: Model or genome file not found - {e}")  # 改为 logging.error (Changed to logging.error)
        input("Press Enter to exit...")
        exit(1)
    except Exception as e:
        logging.error(f"Unknown error occurred while loading models: {e}", exc_info=True)  # exc_info=True 会自动附带 traceback 信息 (exc_info=True automatically includes traceback info)
        input("Press Enter to exit...")
        exit(1)


# --- Flask API 路由 (Flask API Routes) ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收客户端发送的 A 和 B 牌组，返回预测的出牌策略 (move)。
    (Receive A and B card groups from client, return predicted move strategy)
    """
    # 4. 记录每个收到的请求，这是调试的起点 (Log each received request as debugging starting point)
    logging.info("================  New Prediction Request Received  ================")

    if not request.is_json:
        logging.warning("Invalid request format: Content-Type is not application/json")
        return jsonify({"error": "Request must be in JSON format"}), 400

    data = request.get_json()
    # 5. 详细记录收到的数据内容，这是最关键的调试信息 (Log detailed received data - crucial debugging info)
    logging.info(f"Received raw data: {data}")

    A = data.get('A')
    B = data.get('B')

    # ... (输入验证逻辑保持不变) (Input validation logic remains unchanged) ...
    if not A or not B:
        return jsonify({"error": "Request body must contain 'A' and 'B' fields"}), 400
    if not isinstance(A, list) or not isinstance(B, list):
        return jsonify({"error": "'A' and 'B' must be lists"}), 400
    if len(A) != NUM_A or len(B) != NUM_B:
        return jsonify({
            "error": f"Incorrect number of cards. Expected A: {NUM_A}, B: {NUM_B}",
            "received": f"A: {len(A)}, B: {len(B)}"
        }), 400

    try:
        A_batch = [A]
        B_batch = [B]

        logging.info(f"Preparing to call AI model for prediction with card groups A: {A_batch}, B: {B_batch}")

        # --- 调用核心预测函数 (Call core prediction function) ---
        moves_batch = Transformer_predict_batch_plus_GA(
            A_batch=A_batch, B_batch=B_batch, genomeforassist=GENOME_FOR_ASSIST,
            TR_model=TR_MODEL, num_a=NUM_A, num_b=NUM_B, device=device
        )

        predicted_move = moves_batch[0]
        # 6. 记录AI模型的输出结果 (Log AI model output results)
        logging.info(f"Prediction successful! AI recommended move: {predicted_move}")

        # 返回预测结果 (Return prediction results)
        return jsonify({
            "success": True,
            "predicted_move": predicted_move
        })

    except Exception as e:
        # 7. 在发生未知错误时，记录详细的错误信息 (Log detailed error info when unknown errors occur)
        logging.error(f"Critical error occurred during prediction process: {e}", exc_info=True)
        return jsonify({"error": "Internal server error, unable to complete prediction"}), 500


# --- 主程序入口 (Main Program Entry Point) ---
if __name__ == '__main__':
    # 在启动服务器前先加载模型 (Load models before starting server)
    load_models()

    logging.info("Models loaded successfully, preparing to start server...")
    logging.info("Server will listen for requests on http://0.0.0.0:5000")

    serve(app, host='0.0.0.0', port=5000)