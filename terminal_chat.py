import requests
import sys

# Configuration
API_URL = "http://localhost:8000/chat"

def chat_loop():
    print("\n" + "="*50)
    print("🤖 TERMINAL WEATHER BOT (Connected to FastAPI)")
    print("   - Type 'quit' or 'exit' to stop.")
    print("   - Try: 'Weather in London' or 'What is a cyclone?'")
    print("="*50 + "\n")

    while True:
        # 1. Get User Input
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        # 2. Send to Backend
        try:
            # We match the data model: {"query": "..."}
            payload = {"query": user_input}
            
            # Send POST request
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # Raise error for 400/500 codes

            # 3. Parse Response
            data = response.json()
            
            # The backend returns {"response": "..."} or a dict for weather
            bot_reply = data.get("response", "No response received.")
            
            # 4. Print Bot Output
            # If it's a dictionary (like raw weather data), print it prettily
            if isinstance(bot_reply, dict):
                print(f"Bot: {bot_reply}")
            else:
                print(f"Bot: {bot_reply}")

        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to backend. Is 'uvicorn' running?")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)

if __name__ == "__main__":
    chat_loop()