from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3-vl:4b"
    db_path: str = "data/forecaster.duckdb"   # the only config the DB seam needs

settings = Settings()
