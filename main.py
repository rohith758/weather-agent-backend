import logging
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Load Environment Variables
load_dotenv()

# Imports from your src
from src.intent_classifier import classify_intent
from src.handlers import handle_live_weather, handle_theory

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --- CORS SETUP ---
# Added localhost so you can still test locally against the live backend
origins = [
    "https://weather-agent-spxl.vercel.app", # Your Production Vercel Frontend
    "http://localhost:5173",                 # Local Vite Development
    "http://localhost:3000"                  # Alternative Local Development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (Change to your specific Vercel URL later for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- MODELS ---
class ChatRequest(BaseModel):
    query: str

class SummaryRequest(BaseModel):
    messages: list[dict] # List of {"role": "user/bot", "content": "..."}

# --- 🧠 MEMORY STORAGE ---
user_session = {
    "last_city": None
}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    logger.info(f"Received query: {user_query}")

    try:
        intent_data = classify_intent(user_query)
        intent = intent_data.get("intent")
        detected_city = intent_data.get("city")

        # Context Memory Logic
        if detected_city and detected_city != "NULL":
            user_session["last_city"] = detected_city
            final_city = detected_city
        else:
            final_city = user_session["last_city"]

        if intent == "weather":
            if not final_city:
                return {"response": "I couldn't identify a specific city. Which city are you asking about?", "source": "System"}
            
            # Inject memory into query
            enhanced_query = user_query if final_city.lower() in user_query.lower() else f"{user_query} in {final_city}"
            response_text = handle_live_weather(query=enhanced_query, city=final_city)
            return {"response": response_text, "source": "Live Weather API"}
            
        else:
            response = handle_theory(user_query)
            return {"response": response, "source": "Knowledge Base (PDF)"}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary")
async def save_summary(request: SummaryRequest):
    """
    Generates a 2-sentence summary of the session and saves it to a local file.
    """
    if not request.messages or len(request.messages) <= 1:
        return {"summary": "No significant conversation to summarize."}
    
    # 1. Prepare history for the LLM
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in request.messages])
    
    try:
        # 2. Generate a concise, fluff-free summary
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional scribe. Summarize the following chat in exactly 2 sentences. Focus on locations and technical topics mentioned. No preamble."},
                {"role": "user", "content": history_text}
            ],
            temperature=0.3
        )
        summary = response.choices[0].message.content
        
        # 3. Save to a file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("chat_summaries.txt", "a") as f:
            f.write(f"[{timestamp}]\n{summary}\n{'='*30}\n")
            
        logger.info("✅ Conversation summary saved.")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summary Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save summary")

if __name__ == "__main__":
    # RAILWAY UPDATE: Railway dynamically assigns a port. We must catch it here.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)