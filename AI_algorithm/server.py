import os
import sys
import signal
import logging
import  torch
from flask import Flask, request, jsonify
from AI_algorithm.Deep_Neural_Network import DNNpredict
from AI_algorithm.GA import Get_GA_Strategy
from AI_algorithm.tool.tool import load_best_genome, simulate_insertion_tool
from AI_algorithm.Deep_Neural_Network import MovePredictor
# 添加父目录到 Python 搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 配置日志记录，减少日志记录级别以加速
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和 PyInstaller 打包环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 构建相对路径
GENOME_PATH = resource_path(os.path.join('trained', 'best_genome.pkl'))
MODEL_PATH = resource_path(os.path.join('trained', 'move_predictor.pth'))

def load_model_from_memory(model_path, device):
    try:
        logger.info(f"Starting to load model from memory: {model_path}")

        # 读取模型文件到内存
        with open(model_path, 'rb') as f:
            model_data = f.read()

        # 将字节数据包装为 BytesIO 对象
        buffer = io.BytesIO(model_data)
        buffer.seek(0)  # 确保可以 seek

        # 初始化模型实例
        model = MovePredictor(input_size=9)
        logger.info(f"Model initialized: {model}")

        # 加载模型的 state_dict
        state_dict = torch.load(buffer, map_location=device,weights_only=True)  # 从 buffer 加载
        logger.info("State dict loaded successfully")

        # 加载 state_dict 到模型
        model.load_state_dict(state_dict)
        logger.info(f"Model after loading: {model}")

        model.to(device)
        model.eval()  # 设置为评估模式
        logger.info("Model loaded and set to evaluation mode")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

import io

def load_genome_and_model():
    """ 延迟加载基因组和模型 """
    try:
        genome = load_best_genome(GENOME_PATH)
        logger.info("Genome loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load genome: {e}")
        raise

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 读取模型文件到内存
        with open(MODEL_PATH, 'rb') as f:
            model_data = io.BytesIO(f.read())  # 读取文件数据到内存缓冲区

        # 手动构建模型并加载权重
        model = load_model_from_memory(MODEL_PATH, device)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    return genome, model

# 延迟加载模型和基因组
GENOME = None
MODEL = None


def calculate_final_score(A, B, strategy):
    """
    计算给定策略的最终得分。
    """
    current_A = A.copy()
    total_score = 0
    for step in strategy:
        card_index = step[0]
        position = step[1]


        # 确保索引在合法范围内
        if not (0 <= card_index < len(B)):
            raise IndexError(f"card_index {card_index} 超出 B 的范围 (0 - {len(B)-1})")

        score, _, _, _, current_A = simulate_insertion_tool(
            current_A, B[card_index], max(0, min(len(current_A), position))
        )
        total_score += score
    return total_score


app = Flask(__name__)

@app.route('/get_strategy', methods=['POST'])
def get_strategy():
    try:
        global GENOME, MODEL
        if GENOME is None or MODEL is None:
            GENOME, MODEL = load_genome_and_model()

        data = request.get_json(force=True)
        A = data.get('A')
        B = data.get('B')

        if not isinstance(A, list) or not isinstance(B, list):
            return jsonify({"error": "Invalid input format"}), 400

        # 确保 MODEL 是 MovePredictor 实例
        if not isinstance(MODEL, MovePredictor):
            logger.error("MODEL is not a MovePredictor instance")
            return jsonify({"error": "Model loading failed"}), 500

        GA_Strategy = Get_GA_Strategy(GENOME, A, B)
        DNN_Strategy, _ = DNNpredict(A, B, MODEL)

        GA_Final_Score = calculate_final_score(A, B, GA_Strategy)
        DNN_Final_Score = calculate_final_score(A, B, DNN_Strategy)

        response = {
            "GA_Strategy": GA_Strategy,
            "GA_Final_Score": GA_Final_Score,
            "DNN_Strategy": DNN_Strategy,
            "DNN_Final_Score": DNN_Final_Score
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
def shutdown_server(signum, frame):
    """
    关闭服务器。
    """
    logger.info("Shutting down server...")
    sys.exit(0)


if __name__ == '__main__':



    # 注册信号处理函数
    signal.signal(signal.SIGINT, shutdown_server)  # Ctrl+C
    signal.signal(signal.SIGTERM, shutdown_server)  # kill 命令

    logger.info("Starting server...")
    # 使用 Gunicorn 启动服务器
    app.run(debug=False, host='0.0.0.0', port=55666)