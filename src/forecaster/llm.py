from openai import OpenAI
from forecaster.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                default_headers={"X-Title": settings.llm_app_title})


def routing() -> dict:
    """The provider-pin request body, or {} when unpinned.

    OpenRouter is a ROUTER, not a provider: the same slug can be served by different
    backends at different quantizations from one call to the next, which would silently
    mix providers within a benchmark round. Pinning makes the route deterministic.

    Returned as an `extra_body` payload the OpenAI SDK passes through verbatim, so no
    provider SDK enters the codebase and callers never build provider-specific JSON --
    this file stays the only one that knows a provider quirk exists. Empty dict off
    OpenRouter, so the request body is byte-identical to the unpinned path."""
    order = [p.strip() for p in settings.llm_provider_order.split(",") if p.strip()]
    if not order:
        return {}
    body = {"order": order,
            "allow_fallbacks": settings.llm_provider_allow_fallbacks,
            # Refuse any endpoint that cannot honor every parameter we sent. One provider can
            # expose SEVERAL endpoints for the same model at different quantizations, and the
            # cheaper one often lacks tool support -- OpenRouter's default is to DROP an
            # unsupported parameter rather than error, so a route to a tools=False endpoint
            # would silently strip the toolset and present as a model that never calls a tool.
            "require_parameters": True}
    if settings.llm_provider_quantization:
        # Quantization varies across (and within) providers; benchmark scores are not
        # comparable across quantizations, so pin it when the round depends on that.
        body["quantizations"] = [q.strip() for q in
                                 settings.llm_provider_quantization.split(",") if q.strip()]
    return {"provider": body}

# Models whose OpenAI-compatible endpoint accepts video_url content parts (a provider
# extension, NOT standard OpenAI -- most vision models take images only). Substring match on
# the served model id; verify new entries against provider docs before adding.
_VIDEO_MODELS = ("minimax",)


def supports_video(model: str) -> bool:
    """True if the model accepts video_url input (mp4 loops); else send the filmstrip only."""
    m = (model or "").lower()
    return any(v in m for v in _VIDEO_MODELS)
