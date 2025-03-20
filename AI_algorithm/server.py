import os
import sys
import signal
import logging
from flask import Flask, request, jsonify
import torch
from AI_algorithm.Deep_Neural_Network import DNNpredict
from AI_algorithm.GA import Get_GA_Strategy
from AI_algorithm.tool.tool import load_best_genome, simulate_insertion_tool

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


def load_genome_and_model():
    """ 延迟加载基因组和模型 """
    try:
        genome = load_best_genome(GENOME_PATH)
        logger.info("Genome loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load genome: {e}")
        raise

    try:
        model = torch.load(MODEL_PATH)
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
        score, _, _, _, current_A = simulate_insertion_tool(
            current_A, B[card_index], max(1, min(len(current_A), position))
        )
        total_score += score
    return total_score


app = Flask(__name__)


@app.route('/get_strategy', methods=['POST'])
def get_strategy():
    """
    处理 POST 请求，返回基于 GA 和 DNN 的策略及其最终得分。
    """
    logger.info("Received request at /get_strategy")
    try:
        # 延迟加载资源
        global GENOME, MODEL
        if GENOME is None or MODEL is None:
            GENOME, MODEL = load_genome_and_model()

        # 解析输入数据
        data = request.get_json(force=True)
        A = data.get('A')
        B = data.get('B')

        if not isinstance(A, list) or not isinstance(B, list):
            logger.error("Invalid input format")
            return jsonify({"error": "Invalid input format"}), 400

        # 获取基于 GA 和 DNN 的策略
        GA_Strategy = Get_GA_Strategy(GENOME, A, B)
        DNN_Strategy, _ = DNNpredict(A, B, MODEL)

        # 计算最终得分
        GA_Final_Score = calculate_final_score(A, B, GA_Strategy)
        DNN_Final_Score = calculate_final_score(A, B, DNN_Strategy)

        # 返回策略和最终得分
        response = {
            "GA_Strategy": GA_Strategy,
            "GA_Final_Score": GA_Final_Score,
            "DNN_Strategy": DNN_Strategy,
            "DNN_Final_Score": DNN_Final_Score
        }
        logger.info(f"Returning response: {response}")
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
