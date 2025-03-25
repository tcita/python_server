import requests
import json

# 定义服务器地址和端口
SERVER_URL = "http://127.0.0.1:55666/get_strategy"

def test_server():
    """
    测试 Flask 服务器是否可以正常返回 GA 和 DNN 的策略及其最终得分。
    """
    try:
        # 构造测试输入数据
        test_data = {
            "A": [7, 2, 1, 9, 11,5],  # 示例 A 列表
            "B": [13, 7, 8]      # 示例 B 列表
        }

        # 发送 POST 请求
        print(f"Sending POST request to {SERVER_URL}...")
        response = requests.post(SERVER_URL, json=test_data)

        # 检查响应状态码
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return

        # 解析响应内容
        result = response.json()
        print("Received response:")
        print(json.dumps(result, indent=4))

        # 验证返回结果
        required_keys = {"GA_Strategy", "GA_Final_Score", "DNN_Strategy", "DNN_Final_Score"}
        if not required_keys.issubset(result.keys()):
            print("Error: Response is missing required keys.")
            return

        print("Test passed. The server returned valid results.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    test_server()