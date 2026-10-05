import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Determine project root path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    # Gemini API Credentials
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API Key from Google AI Studio")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    
    # Rate Limiter
    GEMINI_RATE_LIMIT_RPM: int = Field(default=12, description="Requests per minute (under 15 RPM free tier)")
    
    # Execution mode
    SIMULATION_MODE: bool = Field(default=False, description="Run offline mock simulation instead of calling live LLM")
    
    # Storage
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./collaborai.db", description="Database connection string")
    
    # Server & Networking
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    ENVIRONMENT: str = Field(default="development")
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
