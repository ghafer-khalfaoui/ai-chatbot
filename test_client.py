import requests

# The URL where your Flask app's /chat endpoint is running
url = "http://127.0.0.1:5000/chat"

# --- CHANGE THIS MESSAGE TO TEST ---
message_data = {
    "message": "help"
}

print(f"Sending message: '{message_data['message']}' to {url}")

try:
    # Send the POST request
    response = requests.post(url, json=message_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the JSON response from the server
        print("Received response:")
        print(response.json())
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:", response.text)

except requests.exceptions.ConnectionError as e:
    print("\n--- CONNECTION ERROR ---")
    print("Could not connect to the server. Is your 'app.py' server running in another terminal?")
    print(f"Details: {e}")