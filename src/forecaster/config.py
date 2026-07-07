from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, from this file's location (src/forecaster/config.py -> up 3), NOT the
# current working directory. Anchoring both paths here means running from any dir
# (or a Slurm job whose cwd is elsewhere) still finds the real .env and DB instead of
# silently falling back to defaults / creating an empty database. Env vars still override.
_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ROOT / ".env")
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3-vl:4b"
    db_path: str = str(_ROOT / "data" / "forecaster.duckdb")   # the only config the DB seam needs

settings = Settings()
