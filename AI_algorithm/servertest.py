import requests
import json
from AI_algorithm.tool.tool import deal_cards_tool

# URL of the server endpoint
SERVER_URL = "http://127.0.0.1:55666/get_strategy"

# Sample input data
A,B=deal_cards_tool()

def test_server():
    """
    测试服务器的 /get_strategy 端点。
    """
    # Prepare the payload
    payload = {
        "A": A,
        "B": B
    }

    # Send a POST request to the server
    response = requests.post(SERVER_URL, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        print("Server Response:")
        print(json.dumps(result, indent=4))

        # Print strategies and their final scores
        print("\nGA Strategy and Final Score:")
        print(f"Strategy: {result['GA_Strategy']} -> Final Score: {result['GA_Final_Score']}")

        print("\nDNN Strategy and Final Score:")
        print(f"Strategy: {result['DNN_Strategy']} -> Final Score: {result['DNN_Final_Score']}")

    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_server()