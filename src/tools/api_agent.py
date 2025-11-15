"""
API Agent Tool - Makes calls to external APIs.

Supports common free APIs:
- Weather (OpenWeatherMap, WeatherAPI)
- Financial (Alpha Vantage, Yahoo Finance)
- Exchange Rates
- News APIs
- Custom REST APIs
"""

import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types."""
    WEATHER = "weather"
    FINANCIAL = "financial"
    EXCHANGE_RATE = "exchange_rate"
    NEWS = "news"
    CUSTOM = "custom"


@dataclass
class APIResponse:
    """Response from API call."""
    success: bool
    data: Any
    status_code: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "status_code": self.status_code,
            "error": self.error
        }


class APIAgentTool:
    """
    Tool for making external API calls.
    
    Usage:
        api_agent = APIAgentTool()
        result = await api_agent.call_weather_api("New York")
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize API agent.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Initialized APIAgentTool")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def call_api(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make a generic API call.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            headers: Request headers
            json_data: JSON body for POST requests
            
        Returns:
            APIResponse with results
        """
        logger.info(f"API call: {method} {url}")
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                
                status = response.status
                
                # Try to parse JSON
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                if 200 <= status < 300:
                    logger.info(f"API call successful: {status}")
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=status
                    )
                else:
                    error = f"HTTP {status}: {data}"
                    logger.error(f"API call failed: {error}")
                    return APIResponse(
                        success=False,
                        data=data,
                        status_code=status,
                        error=error
                    )
                    
        except aiohttp.ClientError as e:
            error = f"Request failed: {str(e)}"
            logger.error(error)
            return APIResponse(success=False, data=None, error=error)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.error(error)
            return APIResponse(success=False, data=None, error=error)
    
    # Weather APIs
    
    async def call_weather_api(
        self,
        location: str,
        api_key: Optional[str] = None,
        provider: str = "weatherapi"
    ) -> APIResponse:
        """
        Get weather information.
        
        Args:
            location: City name or coordinates
            api_key: API key (required)
            provider: "weatherapi" or "openweathermap"
            
        Returns:
            APIResponse with weather data
        """
        if not api_key:
            return APIResponse(
                success=False,
                data=None,
                error="API key required for weather data"
            )
        
        if provider == "weatherapi":
            # WeatherAPI.com (free tier: 1M calls/month)
            url = "http://api.weatherapi.com/v1/current.json"
            params = {"key": api_key, "q": location}
            
        elif provider == "openweathermap":
            # OpenWeatherMap (free tier: 60 calls/minute)
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {"q": location, "appid": api_key, "units": "metric"}
        
        else:
            return APIResponse(
                success=False,
                data=None,
                error=f"Unknown weather provider: {provider}"
            )
        
        return await self.call_api(url, params=params)
    
    # Financial APIs
    
    async def call_stock_api(
        self,
        symbol: str,
        api_key: Optional[str] = None
    ) -> APIResponse:
        """
        Get stock price information.
        
        Uses Alpha Vantage (free tier: 25 calls/day).
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            api_key: Alpha Vantage API key
            
        Returns:
            APIResponse with stock data
        """
        if not api_key:
            # Fallback to Yahoo Finance (no key needed but limited)
            return await self._yahoo_finance(symbol)
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key
        }
        
        return await self.call_api(url, params=params)
    
    async def _yahoo_finance(self, symbol: str) -> APIResponse:
        """Fallback to Yahoo Finance."""
        # Note: This uses an unofficial endpoint
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1d", "range": "1d"}
        
        return await self.call_api(url, params=params)
    
    async def call_exchange_rate_api(
        self,
        base: str = "USD",
        target: Optional[str] = None
    ) -> APIResponse:
        """
        Get currency exchange rates.
        
        Uses exchangerate-api.com (free tier: 1500 calls/month).
        
        Args:
            base: Base currency code (e.g., "USD")
            target: Target currency (optional, returns all if None)
            
        Returns:
            APIResponse with exchange rates
        """
        url = f"https://api.exchangerate-api.com/v4/latest/{base}"
        
        response = await self.call_api(url)
        
        # Filter to target if specified
        if response.success and target and isinstance(response.data, dict):
            rates = response.data.get("rates", {})
            if target in rates:
                response.data = {
                    "base": base,
                    "target": target,
                    "rate": rates[target]
                }
        
        return response
    
    # News APIs
    
    async def call_news_api(
        self,
        query: str,
        api_key: Optional[str] = None,
        country: str = "us"
    ) -> APIResponse:
        """
        Get news articles.
        
        Uses NewsAPI.org (free tier: 100 calls/day).
        
        Args:
            query: Search query
            api_key: NewsAPI key
            country: Country code
            
        Returns:
            APIResponse with news articles
        """
        if not api_key:
            return APIResponse(
                success=False,
                data=None,
                error="API key required for news data"
            )
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 10
        }
        
        return await self.call_api(url, params=params)
    
    # Utility methods
    
    def format_weather(self, response: APIResponse) -> str:
        """Format weather response as readable text."""
        if not response.success:
            return f"Error: {response.error}"
        
        data = response.data
        
        try:
            # WeatherAPI.com format
            if "current" in data:
                current = data["current"]
                location = data["location"]
                return (
                    f"Weather in {location['name']}, {location['country']}:\n"
                    f"Temperature: {current['temp_c']}°C ({current['temp_f']}°F)\n"
                    f"Condition: {current['condition']['text']}\n"
                    f"Humidity: {current['humidity']}%\n"
                    f"Wind: {current['wind_kph']} kph"
                )
            
            # OpenWeatherMap format
            elif "main" in data:
                return (
                    f"Weather in {data['name']}:\n"
                    f"Temperature: {data['main']['temp']}°C\n"
                    f"Condition: {data['weather'][0]['description']}\n"
                    f"Humidity: {data['main']['humidity']}%"
                )
        except KeyError:
            pass
        
        return json.dumps(data, indent=2)
    
    def format_stock(self, response: APIResponse) -> str:
        """Format stock response as readable text."""
        if not response.success:
            return f"Error: {response.error}"
        
        data = response.data
        
        try:
            # Alpha Vantage format
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return (
                    f"Stock: {quote.get('01. symbol', 'N/A')}\n"
                    f"Price: ${quote.get('05. price', 'N/A')}\n"
                    f"Change: {quote.get('09. change', 'N/A')} "
                    f"({quote.get('10. change percent', 'N/A')})"
                )
        except (KeyError, TypeError):
            pass
        
        return json.dumps(data, indent=2)


# Example usage
async def demo():
    """Demonstrate API agent."""
    
    async with APIAgentTool() as api_agent:
        
        print("="*60)
        print("API Agent Demo")
        print("="*60)
        
        # Test 1: Exchange rates (no key needed)
        print("\n1. Exchange Rates (USD to EUR)")
        result = await api_agent.call_exchange_rate_api("USD", "EUR")
        if result.success:
            print(f"   Rate: {result.data['rate']}")
        else:
            print(f"   Error: {result.error}")
        
        # Test 2: Custom API call
        print("\n2. Custom API (Random Dog Image)")
        result = await api_agent.call_api("https://dog.ceo/api/breeds/image/random")
        if result.success:
            print(f"   Image URL: {result.data.get('message', 'N/A')}")
        else:
            print(f"   Error: {result.error}")
        
        # Test 3: Weather (requires API key)
        print("\n3. Weather API (requires key - skipped in demo)")
        print("   Set WEATHER_API_KEY env var to test")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())