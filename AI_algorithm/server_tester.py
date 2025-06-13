import requests
import json

# URL of the server endpoint you created
SERVER_URL = "http://127.0.0.1:5000/predict"


def test_prediction_server():
    """
    Tests the /predict endpoint of the Flask server.
    """
    # Prepare a static, valid payload to ensure the test is repeatable.
    # The server expects A to have 6 cards and B to have 3 cards.
    payload = {
        "A": [2, 4, 6, 8, 10, 12],
        "B": [4, 7, 10]
    }

    print("Sending the following payload to the server:")
    print(json.dumps(payload, indent=4))
    print("-" * 30)

    try:
        # Send a POST request to the server
        response = requests.post(SERVER_URL, json=payload)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            print("✅ Success! Server Response:")
            print(json.dumps(result, indent=4))

            # Extract and display the predicted strategy
            if result.get("success"):
                predicted_move = result.get("predicted_move")
                print("\nPredicted Strategy (move):")
                print(predicted_move)
            else:
                print("\n⚠️ Server reported success=false.")

        # Handle other common HTTP errors
        elif response.status_code == 400:
            print(f"❌ Error: Bad Request (400). The server rejected the input.")
            print("Server message:", response.text)
        elif response.status_code == 500:
            print(f"❌ Error: Internal Server Error (500).")
            print("Server message:", response.text)
        else:
            print(f"❌ An unexpected error occurred. Status Code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: Could not connect to the server at {SERVER_URL}.")
        print("Please ensure the server.py script is running.")
    except Exception as e:
        print(f"An unexpected error occurred during the request: {e}")


if __name__ == "__main__":
    test_prediction_server()