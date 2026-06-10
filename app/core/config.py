from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    SERPER_API_KEY: str = ""
    SEARCH_PROVIDER: str = "duckduckgo" # Can be 'duckduckgo' or 'serper'
    DATABASE_URL: str = "sqlite:///./tasks.db"
    WEBHOOK_URL: Optional[str] = None
    MAX_CONCURRENT_BROWSERS: int = 4
    PDF_MAX_FILE_SIZE_MB: int = 20  # Maximum PDF upload size for /extract-pdf/

    # LLM model names — override via .env to switch models without code changes.
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_OCR_MODEL: str = "gpt-4o-mini"  # Model used for PDF vision/OCR processing
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_OCR_MODEL: str = "gemini-2.5-flash-lite"  # Model used for PDF native/Files API OCR processing
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Loads .env file
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings = Settings()
