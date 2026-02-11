import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)


class WeatherQueryType(Enum):
    """Enumeration for different types of weather queries."""
    CURRENT = 1
    FORECAST = 5
    EXTENDED = 7


class WeatherAPIError(Exception):
    """Custom exception for weather API errors."""
    pass


class WeatherAPIClient:
    """
    Client for fetching and processing weather data with LLM integration.
    
    This client:
    1. Uses LLM to extract and correct city names from  language queries
    2. Fetches weather data from WeatherAPI.comnatural
    3. Uses LLM to generate natural, conversational responses
    """
    
    # API Configuration
    BASE_URL = "http://api.weatherapi.com/v1/forecast.json"
    REQUEST_TIMEOUT = 10  # seconds
    DEFAULT_DAYS = 1
    MAX_FORECAST_DAYS = 7
    
    # LLM Configuration
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0
    
    # System prompts
    EXTRACTION_SYSTEM_PROMPT = """
You are an intelligent entity extractor for weather queries.

TASKS:
1. Identify the 'city' from the user's query
2. CORRECT any spelling errors to standard English city names:
   - 'Madhurai' → 'Madurai'
   - 'Banglore' → 'Bengaluru'
   - 'Newyork' → 'New York'
   - 'Dilli' → 'Delhi'
3. Determine 'days' based on query context:
   - Words like 'forecast', 'week', 'coming days', 'next few days' → days = 3
   - Words like 'today', 'current', 'now' → days = 1
   - No time reference → days = 1
4. Return JSON: {"city": "CorrectedCityName", "days": integer}
5. If no city is found, set city to null

EXAMPLES:
- "weather in banglore next week" → {"city": "Bengaluru", "days": 3}
- "temperature in paris" → {"city": "Paris", "days": 1}
- "forecast for newyork" → {"city": "New York", "days": 3}
"""

    RESPONSE_SYSTEM_PROMPT = """
You are a friendly Weather Assistant.

GUIDELINES:
- Use the provided JSON weather data to answer the user's specific question
- For activity-based questions (e.g., "Can I play cricket?", "Should I carry an umbrella?"):
  * Analyze relevant weather factors (rain, wind, temperature)
  * Provide actionable advice
- Keep responses concise, friendly, and use emojis appropriately
- Write in natural language - DO NOT mention "JSON", "data", or technical terms
- Focus on what matters to the user based on their query
- Use temperature in Celsius by default

RESPONSE STYLE:
- Current weather: Brief and informative
- Forecast: Highlight key changes or notable conditions
- Activity advice: Clear recommendation with reasoning
"""

    def __init__(
        self,
        weather_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the WeatherAPIClient.
        
        Args:
            weather_api_key: WeatherAPI.com API key. If None, loads from environment.
            openai_api_key: OpenAI API key. If None, loads from environment.
            base_url: Weather API base URL. If None, uses default.
            
        Raises:
            ValueError: If required API keys are missing.
        """
        # Load API keys
        self.weather_api_key = weather_api_key or os.getenv("WEATHER_API_KEY")
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Validate API keys
        if not self.weather_api_key:
            raise ValueError(
                "❌ WEATHER_API_KEY is missing. "
                "Please set it in .env file or pass as argument."
            )
        
        if not openai_key:
            raise ValueError(
                "❌ OPENAI_API_KEY is missing. "
                "Please set it in .env file or pass as argument."
            )
        
        # Initialize clients
        self.base_url = base_url or self.BASE_URL
        self.openai_client = OpenAI(api_key=openai_key)
        
        logger.info("✅ WeatherAPIClient initialized successfully")

    def _extract_location_and_days(self, user_query: str) -> Tuple[Optional[str], int]:
        """
        Extract city name and forecast days from user query using LLM.
        
        Args:
            user_query: Natural language query from user.
            
        Returns:
            Tuple of (city_name, days) where city_name can be None.
            
        Raises:
            WeatherAPIError: If extraction fails.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=self.LLM_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            city = extracted_data.get("city")
            days = extracted_data.get("days", self.DEFAULT_DAYS)
            
            # Validate and normalize
            if city and city.upper() == "NULL":
                city = None
            
            # Clamp days to valid range
            days = max(1, min(days, self.MAX_FORECAST_DAYS))
            
            logger.info(f"Extracted from '{user_query}': city={city}, days={days}")
            return city, days
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM extraction response: {e}")
            raise WeatherAPIError("Failed to process your query") from e
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            raise WeatherAPIError("Failed to understand your query") from e

    def _fetch_raw_weather(self, location: str, days: int = 1) -> Dict[str, Any]:
        """
        Fetch raw weather data from WeatherAPI.com.
        
        Args:
            location: City name or location query.
            days: Number of forecast days (1-7).
            
        Returns:
            Dictionary with 'success' key and either 'data' or 'error'.
            
        Raises:
            WeatherAPIError: If API request fails critically.
        """
        try:
            params = {
                "key": self.weather_api_key,
                "q": location,
                "days": days,
                "aqi": "no",
                "alerts": "no"
            }
            
            logger.info(f"Fetching weather for {location} ({days} days)")
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.REQUEST_TIMEOUT
            )
            
            # Handle specific error cases
            if response.status_code == 400:
                logger.warning(f"City not found: {location}")
                return {"success": False, "error": "City not found"}
            
            if response.status_code == 401:
                logger.error("Invalid Weather API key")
                return {"success": False, "error": "API authentication failed"}
            
            if response.status_code == 403:
                logger.error("Weather API access forbidden")
                return {"success": False, "error": "API access denied"}
            
            response.raise_for_status()
            raw_data = response.json()
            
            # Transform to clean structure
            cleaned_data = self._clean_weather_data(raw_data)
            
            logger.info(f"Successfully fetched weather for {location}")
            return {"success": True, "data": cleaned_data}
            
        except requests.Timeout:
            logger.error(f"Request timeout for location: {location}")
            return {"success": False, "error": "Request timed out"}
        except requests.RequestException as e:
            logger.error(f"Weather API request error: {e}")
            return {"success": False, "error": "Failed to fetch weather data"}
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            return {"success": False, "error": "An unexpected error occurred"}

    def _clean_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and structure raw weather API response to reduce token usage.
        
        Args:
            raw_data: Raw JSON response from WeatherAPI.com.
            
        Returns:
            Cleaned and structured weather data.
        """
        location_data = raw_data.get("location", {})
        current_data = raw_data.get("current", {})
        forecast_data = raw_data.get("forecast", {}).get("forecastday", [])
        
        cleaned = {
            "location": f"{location_data.get('name', 'Unknown')}, "
                       f"{location_data.get('country', 'Unknown')}",
            "local_time": location_data.get("localtime", "N/A"),
            "current": {
                "temp": f"{current_data.get('temp_c', 'N/A')}°C",
                "feels_like": f"{current_data.get('feelslike_c', 'N/A')}°C",
                "condition": current_data.get("condition", {}).get("text", "N/A"),
                "wind": f"{current_data.get('wind_kph', 'N/A')} kph",
                "wind_direction": current_data.get("wind_dir", "N/A"),
                "humidity": f"{current_data.get('humidity', 'N/A')}%",
                "cloud_cover": f"{current_data.get('cloud', 'N/A')}%",
                "uv_index": current_data.get("uv", "N/A")
            },
            "forecast": []
        }
        
        # Add forecast data
        for day in forecast_data:
            day_data = day.get("day", {})
            cleaned["forecast"].append({
                "date": day.get("date", "N/A"),
                "max_temp": f"{day_data.get('maxtemp_c', 'N/A')}°C",
                "min_temp": f"{day_data.get('mintemp_c', 'N/A')}°C",
                "avg_temp": f"{day_data.get('avgtemp_c', 'N/A')}°C",
                "condition": day_data.get("condition", {}).get("text", "N/A"),
                "rain_chance": f"{day_data.get('daily_chance_of_rain', 'N/A')}%",
                "max_wind": f"{day_data.get('maxwind_kph', 'N/A')} kph"
            })
        
        return cleaned

    def _generate_conversational_response(
        self,
        user_query: str,
        weather_data: Dict[str, Any]
    ) -> str:
        """
        Generate a natural language response using LLM.
        
        Args:
            user_query: Original user query.
            weather_data: Cleaned weather data.
            
        Returns:
            Natural language response string.
        """
        try:
            prompt = (
                f"User Query: {user_query}\n\n"
                f"Weather Data: {json.dumps(weather_data, indent=2)}"
            )
            
            response = self.openai_client.chat.completions.create(
                model=self.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {e}")
            # Fallback to basic response
            current = weather_data.get("current", {})
            location = weather_data.get("location", "Unknown")
            temp = current.get("temp", "N/A")
            condition = current.get("condition", "N/A")
            return f"Weather in {location}: {temp}, {condition}."

    def get_weather(self, user_query: str) -> str:
        """
        Process a natural language weather query and return a response.
        
        This is the main entry point that orchestrates:
        1. Entity extraction (city, days)
        2. Weather data fetching
        3. Natural language response generation
        
        Args:
            user_query: Natural language query from user.
            
        Returns:
            Natural language response string.
        """
        if not user_query or not user_query.strip():
            return "Please ask me about the weather in a specific location."
        
        user_query = user_query.strip()
        
        # Phase 1: Extract location and days
        try:
            city, days = self._extract_location_and_days(user_query)
        except WeatherAPIError as e:
            logger.error(f"Extraction failed: {e}")
            return "I'm having trouble understanding your query. Could you rephrase it?"
        
        # Validate city was extracted
        if not city:
            return (
                "I couldn't identify a city in your request. "
                "Could you please specify the location? "
                "For example: 'weather in London' or 'forecast for Paris'"
            )
        
        # Phase 2: Fetch weather data
        weather_result = self._fetch_raw_weather(city, days)
        
        if not weather_result.get("success"):
            error_msg = weather_result.get("error", "Unknown error")
            
            if "not found" in error_msg.lower():
                return (
                    f"I couldn't find weather data for '{city}'. "
                    f"Please check the spelling or try a different location."
                )
            elif "authentication" in error_msg.lower() or "denied" in error_msg.lower():
                return "I'm experiencing technical difficulties. Please try again later."
            elif "timeout" in error_msg.lower():
                return "The weather service is taking too long to respond. Please try again."
            else:
                return f"I encountered an issue: {error_msg}. Please try again."
        
        # Phase 3: Generate conversational response
        try:
            response = self._generate_conversational_response(
                user_query,
                weather_result["data"]
            )
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Return basic weather info as fallback
            data = weather_result["data"]
            location = data.get("location", "Unknown")
            current = data.get("current", {})
            return f"Weather in {location}: {current.get('temp', 'N/A')}, {current.get('condition', 'N/A')}."


# Convenience function for simple usage
def get_weather_response(query: str) -> str:
    """
    Convenience function to get weather response for a query.
    
    Args:
        query: Natural language weather query.
        
    Returns:
        Natural language response.
    """
    try:
        client = WeatherAPIClient()
        return client.get_weather(query)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return "Weather service is not properly configured."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again later."


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test queries
    test_queries = [
        "weather in mangalore",
        "forecast for banglore next week",
        "will it rain in paris tomorrow",
        "temperature in newyork",
        "can I play cricket in London?",
        "what about delhi",  # Missing context - should prompt for location
    ]
    
    try:
        client = WeatherAPIClient()
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            response = client.get_weather(query)
            print(f"Response: {response}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure WEATHER_API_KEY and OPENAI_API_KEY are set in .env file")