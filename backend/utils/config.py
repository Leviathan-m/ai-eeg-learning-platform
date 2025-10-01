"""
Configuration management for AI-EEG Learning Platform.

Uses Pydantic settings for type-safe configuration with environment variable support.
"""

import secrets
from typing import List, Optional, Union

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """

    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Server Configuration
    SERVER_NAME: str = "AI-EEG Learning Platform"
    SERVER_HOST: str = "http://localhost"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Alternative frontend port
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from environment variable."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        elif isinstance(v, str):
            # Leave JSON-style string list parsing to Pydantic if provided
            # but ensure type consistency for mypy by returning a list
            return [v]
        raise ValueError(v)

    # Trusted Hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    # Database Configuration
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/eeg_learning"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # EEG Processing Configuration
    EEG_SAMPLING_RATE: int = 256  # Hz
    EEG_CHANNELS: int = 4
    EEG_PROCESSING_THREADS: int = 4
    EEG_BUFFER_SIZE: int = 1024
    EEG_MAX_LATENCY_MS: int = 100

    # Machine Learning Configuration
    ML_MODEL_PATH: str = "/app/ml_models"
    ML_BATCH_SIZE: int = 32
    ML_MAX_SEQUENCE_LENGTH: int = 100
    ML_EMBEDDING_DIM: int = 128

    # Hardware Integration
    MUSE_AUTO_DETECT: bool = True
    EMOTIV_CLIENT_ID: Optional[str] = None
    EMOTIV_CLIENT_SECRET: Optional[str] = None

    # Security Configuration
    ENCRYPTION_KEY: str = secrets.token_hex(32)  # AES-256 key
    JWT_ALGORITHM: str = "HS256"

    # Monitoring and Logging
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Performance Tuning
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 30
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # External Services
    EMAIL_ENABLED: bool = False
    EMAIL_SERVER: Optional[str] = None
    EMAIL_PORT: Optional[int] = None
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None

    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".json", ".edf"]

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
