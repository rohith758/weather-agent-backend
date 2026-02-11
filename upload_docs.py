import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

def upload_and_create_store():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)

    # 1. Create the Store
    print("📦 Creating File Search Store...")
    file_search_store = client.file_search_stores.create(
        config={'display_name': 'weather_store_v2'}
    )
    print(f"✅ Store Created: {file_search_store.name}")

    # 2. Locate Docs
    # We check both locations to be safe
    possible_folders = ["backend/docs", "docs"]
    docs_folder = next((f for f in possible_folders if os.path.exists(f)), None)

    if not docs_folder:
        print("❌ Error: Cannot find 'docs' folder. Please create it and add PDFs.")
        return

    files_to_upload = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
    
    if not files_to_upload:
        print(f"⚠️ No PDF files found in '{docs_folder}' folder.")
        return

    # 3. Upload Files with Metadata (Optional Fine-tuning)
    for file_name in files_to_upload:
        file_path = os.path.join(docs_folder, file_name)
        print(f"Indexing {file_name}...")
        
        # Adding metadata helps the AI narrow down searches later
        custom_metadata = [
            {'key': 'category', 'string_value': 'climate_science'},
            {'key': 'filename', 'string_value': file_name}
        ]

        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_path,  
            file_search_store_name=file_search_store.name,
            config={
                'display_name': file_name,
                'custom_metadata': custom_metadata # This is a retrieval "fine-tune"
            }
        )
        # Wait for completion
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
            print(".", end="", flush=True)
        print(f" Done!")

    print("\n🎉 SUCCESS!")
    print(f"👉 Copy this to your .env file:\nGEMINI_STORE_ID={file_search_store.name}")

if __name__ == "__main__":
    upload_and_create_store()