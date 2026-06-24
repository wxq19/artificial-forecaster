import base64
from forecaster.llm import client
from forecaster.config import settings

print("=== TEXT CALL ===")
r = client.chat.completions.create(
    model=settings.llm_model,
    messages=[{"role": "user", "content": "Reply with exactly: text works"}],
)
print(r.choices[0].message.content)

print("\n=== VISION CALL (starting, may be slow) ===")
with open("data/charts/nws/test.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
print(f"Image loaded: {len(b64)} base64 chars")

r = client.chat.completions.create(
    model=settings.llm_model,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
)
print(r.choices[0].message.content)
print("\n=== DONE ===")
