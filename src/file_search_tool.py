import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiFileSearch:
    """
    Smart Knowledge Base Tool:
    1. Handles Greetings LOCALLY (Zero API Cost).
    2. Uses 'gemini-3-flash-preview' for RAG (Newer model, separate quota).
    3. Falls back to Chat Mode if RAG fails.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.store_id = os.getenv("GEMINI_STORE_ID")
        
        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY is missing")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # ✅ SWITCHED to a Preview model to bypass current rate limits
        # You do NOT need to recreate your database/store.
        self.model_id = "gemini-3-flash-preview"

    def search(self, query: str):
        """
        Main function to handle Theory/Concepts.
        """
        # --- ⚡️ LEVEL 1: ZERO-COST LOCAL GREETINGS ---
        greetings = ["hi", "hello", "hey", "hola", "namaste", "thanks", "thank you", "bye", "good morning"]
        cleaned_query = query.strip().lower().replace("!", "").replace(".", "")
        
        if cleaned_query in greetings:
            logger.info(f"⚡ Handling greeting locally: {query}")
            return "Hello! 👋 I am your Weather Intelligence Assistant. Ask me for a **Forecast** (e.g., 'London Weather') or a **Concept** (e.g., 'What is a cyclone?')."

        # --- 🔍 LEVEL 2: INTELLIGENT SEARCH ---
        try:
            # Try Document Search (RAG)
            if self.store_id:
                logger.info(f"🔍 RAG Search for: '{query}' using {self.model_id}")
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=query,
                        config=types.GenerateContentConfig(
                            tools=[
                                types.Tool(
                                    file_search=types.FileSearch(
                                        file_search_store_names=[self.store_id]
                                    )
                                )
                            ]
                        )
                    )
                    return response.text
                except Exception as rag_error:
                    logger.warning(f"⚠️ RAG Search failed ({rag_error}). Switching to Chat Mode.")
            
            # --- 💬 LEVEL 3: FALLBACK CHAT ---
            logger.info(f"💬 Chat Mode for: '{query}'")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"You are a helpful weather assistant. User says: {query}",
                config=types.GenerateContentConfig()
            )
            return response.text

        except Exception as e:
            logger.error(f"❌ Gemini Critical Error: {e}")
            if "429" in str(e):
                return "⚠️ **System Overload:** I'm receiving too many requests right now. Please wait 1 minute and try again!"
            return "I'm having trouble connecting to my knowledge base right now. Please try again later."