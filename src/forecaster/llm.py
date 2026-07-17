from openai import OpenAI
from forecaster.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

# Models whose OpenAI-compatible endpoint accepts video_url content parts (a provider
# extension, NOT standard OpenAI -- most vision models take images only). Substring match on
# the served model id; verify new entries against provider docs before adding.
_VIDEO_MODELS = ("minimax",)


def supports_video(model: str) -> bool:
    """True if the model accepts video_url input (mp4 loops); else send the filmstrip only."""
    m = (model or "").lower()
    return any(v in m for v in _VIDEO_MODELS)
