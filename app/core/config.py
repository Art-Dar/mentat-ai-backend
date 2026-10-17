from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Second Brain API"
    environment: str = "local"  # local/staging/production
    debug: bool = True
    database_url: str

    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week — two-user app, long-lived is fine
    voyage_api_key: SecretStr
    embedding_model: str = "voyage-4"
    google_client_id: str = ""
    groq_api_key: SecretStr = ""
    tagging_model: str = "openai/gpt-oss-20b"
    storage_bucket: str = "mentat-dev"
    storage_endpoint_url: str = ""
    storage_access_key: str = ""
    storage_secret_key: SecretStr = SecretStr("")
    storage_region: str = "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
