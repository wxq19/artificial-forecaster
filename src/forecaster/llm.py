from openai import OpenAI
from forecaster.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
