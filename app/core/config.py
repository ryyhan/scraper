from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None
    DATABASE_URL: str = "sqlite:///./tasks.db"
    WEBHOOK_URL: Optional[str] = None
    MAX_CONCURRENT_BROWSERS: int = 4
    
    # Loads .env file
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings = Settings()
