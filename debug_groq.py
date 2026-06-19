import httpx
import os

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}
payload = {
    "model": "llama3-8b-8192",
    "messages": [
        {"role": "user", "content": "hi"}
    ]
}

res = httpx.post(url, headers=headers, json=payload)
print(res.status_code)
print(res.text)
