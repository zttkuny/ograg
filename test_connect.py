import os
from openai import OpenAI
import httpx

api_key = "sk-xwhmJ0SnByTG9D9cjYJVvv19IgKUd9RqK41AO3mOu3lGWnXL"
base_url = "https://api.openai-proxy.org/v1"

print(f"Testing connection to {base_url} with key {api_key[:5]}...")

try:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False) # Disable SSL verify just in case
    )

    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-small"
    )
    print("Embeddings Success!")
    print(response.data[0].embedding[:5])
except Exception as e:
    print("Embeddings Error:")
    print(e)

try:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="deepseek-chat"
    )
    print("Chat Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print("Chat Error:")
    print(e)