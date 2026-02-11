import logging
import os
from openai import OpenAI
from src.weather_api_client import WeatherAPIClient
from src.file_search_tool import GeminiFileSearch

logger = logging.getLogger(__name__)

# Initialize Clients
weather_agent = WeatherAPIClient()
gemini_tool = GeminiFileSearch()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handle_live_weather(query: str = "", city: str = None):
    input_text = query if query else str(city)
    return weather_agent.get_weather(input_text)

def handle_theory(user_query: str):
    # 1. Local Greeting Check (The Fix)
    greetings = ["hi", "hello", "hey", "hola"]
    if user_query.lower().strip() in greetings:
        return "Hello! I'm your Weather Intelligence Bot. Ask me about a city's forecast or a climate concept from your docs."

    # 2. Proceed to PDF Search if not a greeting
    raw_knowledge = gemini_tool.search(user_query) 
    
    if not raw_knowledge:
        return "I couldn't find any information on that topic in my knowledge base."

    return generate_summarized_response(user_query, [raw_knowledge])

# ... (existing imports)

def generate_summarized_response(user_query, retrieved_chunks):
    """
    Synthesizes a response grounded strictly in the retrieved PDF context.
    """
    raw_context = "\n---\n".join(retrieved_chunks)
    
    # IMPROVED SYSTEM PROMPT: Focuses on grounding and avoiding hallucinations
    system_prompt = (
        "You are a Weather Intelligence Expert. Your goal is to answer questions "
        "using ONLY the provided context from research documents. \n\n"
        "RULES:\n"
        "1. If the answer is NOT in the context, say: 'I'm sorry, my current documents "
        "do not contain information on that specific topic.'\n"
        "2. Do not use outside knowledge.\n"
        "3. Keep the tone professional and the answer concise (under 4 sentences).\n"
        "4. Do not mention 'chunks' or 'files'."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Use mini for faster, cheaper synthesis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{raw_context}\n\nUSER QUESTION: {user_query}"}
            ],
            temperature=0.2 # Lower temperature = higher factual accuracy
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Synthesis Error: {e}")
        return "I had trouble processing that document search."