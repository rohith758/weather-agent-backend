import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class IntentClassifier:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def classify(self, query: str) -> dict:
        # FEW-SHOT EXAMPLES: Teach the model by showing, not just telling
        examples = """
        Query: "What's the rain like in Seattle?" -> {"intent": "weather", "city": "Seattle"}
        Query: "How do cyclones form?" -> {"intent": "theory", "city": null}
        Query: "Will it be sunny tomorrow?" -> {"intent": "weather", "city": null}
        Query: "Explain humidity according to the docs" -> {"intent": "theory", "city": null}
        """

        prompt = f"""
        You are the Routing Brain for a Weather AI. Analyze the user query and determine if they want LIVE weather data or THEORETICAL knowledge from a PDF.

        EXAMPLES:
        {examples}

        USER QUERY: "{query}"

        RULES:
        1. "weather": Use for current/future conditions, forecasts, or city-specific weather checks.
        2. "theory": Use for scientific definitions, greetings, or "how it works" questions.
        3. "city": Extract the city name if present; otherwise, return null.

        Output valid JSON only.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise JSON classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0, # 0 is best for consistent classification
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"intent": "theory", "city": None}

def classify_intent(query: str) -> dict:
    return IntentClassifier().classify(query)