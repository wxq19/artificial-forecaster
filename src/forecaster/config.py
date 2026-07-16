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
    # One fixed seed across the whole benchmark matrix, so temperature (0 vs 0.2) is the
    # only varied sampling knob and any config cell is reproducible in principle. Sent to
    # the API only when set (best-effort determinism on Together; None = omit the param).
    llm_seed: int | None = 1337
    db_path: str = str(_ROOT / "data" / "forecaster.duckdb")   # the only config the DB seam needs
    # Climatology period-of-record. end_year is the last COMPLETE year: a historical
    # valid-time run can't absorb post-cutoff obs through the climo product (leakage guard).
    climo_start_year: int = 2006
    climo_end_year: int = 2025
    # TAF worksheet (docs/taf_worksheet_design.md). The agent fills a structured
    # reasoning worksheet before emit_taf; these govern whether it is requested/gated.
    #   off      - no worksheet (experiment control arm)
    #   advisory - worksheet validated + findings surfaced; never blocks emit_taf (default)
    #   required - worksheet must pass semantic validation before emit_taf is accepted
    worksheet_mode: str = "advisory"
    #   off | key_claims (default) | strict -- how strictly evidence_refs are demanded
    evidence_mode: str = "key_claims"
    # Persist the final accepted worksheet + evidence + TAF + findings to the store.
    persist_worksheets: bool = True
    # get_previous_taf leakage guard: the "last available TAF" fed to the model must have
    # been issued at least this many minutes BEFORE the collection cutoff, so an early-posted
    # current-cycle TAF can never leak into the model's context.
    previous_taf_buffer_min: int = 15

settings = Settings()
