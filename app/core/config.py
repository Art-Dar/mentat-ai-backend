from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Second Brain API"
    environment: str = "local"   # local/staging/production
    debug: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()