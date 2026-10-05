"""CollaborAI LLM Layer.

Provides modular LLM access with Gemini Free Tier rate-limiting safeguards
and a deterministic simulation provider for offline/zero-token operation.
"""

from .base import BaseLLMProvider
from .rate_limiter import rate_limiter, TokenBucketRateLimiter
from .gemini import GeminiProvider
from .mock_provider import MockLLMProvider
from ..config import settings


def get_llm_provider(force_simulation: bool = False) -> BaseLLMProvider:
    """Factory providing the configured LLM client.
    
    Returns GeminiProvider if API key is present and simulation mode is False,
    otherwise returns MockLLMProvider.
    """
    if force_simulation or settings.SIMULATION_MODE or not settings.GEMINI_API_KEY:
        return MockLLMProvider()
    return GeminiProvider()


__all__ = [
    "BaseLLMProvider",
    "rate_limiter",
    "TokenBucketRateLimiter",
    "GeminiProvider",
    "MockLLMProvider",
    "get_llm_provider",
]
