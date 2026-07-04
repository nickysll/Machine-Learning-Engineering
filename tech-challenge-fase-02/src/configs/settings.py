from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    mlflow_tracking_uri: str = "http://localhost:5000"
    dvc_remote: str = "local"


settings = Settings()
