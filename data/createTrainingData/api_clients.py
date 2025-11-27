"""
api_clients.py

API client implementations for various LLM providers.
Each client handles rate limiting and provider-specific quirks.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional


class BaseAPIClient(ABC):
    """Base class for all API clients"""
    
    def __init__(self, api_key: str, model: str, requests_per_second: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.requests_per_second = requests_per_second
        self.delay_between_requests = 1.0 / requests_per_second
        self.request_count = 0
        self.last_request_time = datetime.now()
    
    async def rate_limit(self):
        """Implement rate limiting"""
        self.request_count += 1
        if self.request_count % 100 == 0:
            print(f"   📊 {self.__class__.__name__}: {self.request_count} requests")
        
        await asyncio.sleep(self.delay_between_requests)
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat request to API"""
        pass
    
    async def query(self, messages: List[Dict[str, str]], retry_count: int = 3) -> str:
        """
        Query API with rate limiting and error handling.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            retry_count: Number of retries on failure
        
        Returns:
            Response text or empty string on failure
        """
        await self.rate_limit()
        
        for attempt in range(retry_count):
            try:
                response = await self.chat(messages)
                return response
            except Exception as e:
                print(f"⚠️  {self.__class__.__name__} error (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    return ""
        
        return ""


class MistralClient(BaseAPIClient):
    """Mistral AI API client"""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            model="mistral-large-2411",
            requests_per_second=1.0  # Free tier: 1 req/sec
        )
        
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
        except ImportError:
            raise ImportError("Please install mistralai: pip install mistralai")
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat request to Mistral"""
        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content


class TogetherClient(BaseAPIClient):
    """Together AI API client"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        super().__init__(
            api_key=api_key,
            model=model,
            requests_per_second=1.0  # Conservative for free tier
        )
        
        try:
            from together import Together
            self.client = Together(api_key=api_key)
        except ImportError:
            raise ImportError("Please install together: pip install together")
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat request to Together AI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content


class HuggingFaceClient(BaseAPIClient):
    """Hugging Face Inference API client"""
    
    def __init__(self, api_key: str, model: str = "Qwen/QwQ-32B-Preview"):
        super().__init__(
            api_key=api_key,
            model=model,
            requests_per_second=0.5  # Conservative for free tier
        )
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=api_key)
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat request to Hugging Face"""
        # HF has a different format
        response = self.client.chat_completion(
            messages=messages,
            model=self.model,
            max_tokens=4000
        )
        return response.choices[0].message.content


# Client factory
def create_client(provider: str, api_key: str, **kwargs) -> BaseAPIClient:
    """
    Factory function to create API clients.
    
    Args:
        provider: Provider name ('mistral', 'together', 'huggingface')
        api_key: API key for the provider
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Configured API client
    """
    clients = {
        'mistral': MistralClient,
        'together': TogetherClient,
        'huggingface': HuggingFaceClient,
    }
    
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(clients.keys())}")
    
    return clients[provider](api_key, **kwargs)