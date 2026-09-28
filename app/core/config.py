from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Second Brain API"
    environment: str = "local"  # local/staging/production
    debug: bool = True
    database_url: str

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week — two-user app, long-lived is fine

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
