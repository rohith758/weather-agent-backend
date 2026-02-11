import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env from backend folder
load_dotenv("backend/.env")

def debug_gemini():
    print("🔍 DIAGNOSTIC TOOL: Gemini File Search")
    print("--------------------------------------")
    
    api_key = os.getenv("GEMINI_API_KEY")
    store_id = os.getenv("GEMINI_STORE_ID")
    
    if not api_key or not store_id:
        print("❌ CRITICAL: Missing API Key or Store ID")
        return

    client = genai.Client(api_key=api_key)
    
    # FIX: Use a supported model from your list
    model_id = "gemini-2.5-flash-lite" 

    print(f"🚀 Attempting Search with model: {model_id}...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="What is climate?",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_id]
                        )
                    )
                ]
            )
        )
        print("\n✅ SUCCESS! Response from Gemini:")
        print(response.text)
        
    except Exception as e:
        print("\n❌ SEARCH FAILED!")
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_gemini()