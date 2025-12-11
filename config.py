from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Google Gemini
    GOOGLE_API_KEY: str
    GEMINI_MODEL: Optional[str] = "gemini-2.5-flash"
    
    # LLaMAParse
    LLAMAPARSE_API_KEY: str
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Memgraph
    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields like Redis settings

settings = Settings()
