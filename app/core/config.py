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
    MAX_CONCURRENT_COMBINED_SEARCHES: int = 5
    PDF_MAX_FILE_SIZE_MB: int = 20  # Maximum PDF upload size for /extract-pdf/
    TRUSTED_PROXIES: str = "127.0.0.1,::1,testclient"  # Comma-separated list or '*'

    # LLM model names — override via .env to switch models without code changes.
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_OCR_MODEL: str = "gpt-4o-mini"  # Model used for PDF vision/OCR processing
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_OCR_MODEL: str = "gemini-2.5-flash-lite"  # Model used for PDF native/Files API OCR processing
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Local audio transcription (/transcribe-audio/) — runs fully on-host via onnx-asr.
    TRANSCRIBE_MODEL: str = "nemo-parakeet-tdt-0.6b-v3"
    TRANSCRIBE_QUANTIZATION: str = "int8"          # "int8" (CPU-friendly) or "none" for full precision
    TRANSCRIBE_MODELS_DIR: str = "models"          # Model dir; relative paths resolve from the project root
    TRANSCRIBE_NUM_THREADS: int = 0                # ONNX intra-op threads; 0 = all cores
    TRANSCRIBE_MAX_SEGMENT_SECONDS: float = 20.0   # VAD max speech segment (Parakeet limit ~20–30 s)
    TRANSCRIBE_PRELOAD_MODEL: bool = True          # Load the model at startup instead of first request
    MAX_CONCURRENT_TRANSCRIPTIONS: int = 1         # CPU-bound; >1 rarely helps on a shared host
    AUDIO_MAX_FILE_SIZE_MB: int = 100

    # Loads .env file
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings = Settings()
