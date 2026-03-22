from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Hospital 2 API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002  # Changed to 8002 to avoid conflicting with central server on 8000
    
    # Central Server Info (useful for node communication)
    CENTRAL_SERVER_URL: str = "https://central-server-production-b21a.up.railway.app"

    # Hospital-2 local backend API (FastAPI)
    LOCAL_API_BASE_URL: str = "https://hospital2-qbao.onrender.com"
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = 8503  # Changed to 8503 to avoid conflicting with central server
    
    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]
    
    # Database (if needed)
    DATABASE_URL: Optional[str] = None
    
    # ML Model Settings
    MODEL_PATH: str = "models/"
    DEFAULT_MODEL: str = "local_model"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
