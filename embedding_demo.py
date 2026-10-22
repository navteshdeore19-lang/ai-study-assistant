
import requests

url = "http://localhost:11434/api/embed"

data = {
    "model": "nomic-embed-text",
    "input": "Max pooling keeps the largest value in each window."
}

response = requests.post(url, json=data)
response.raise_for_status()

result = response.json()

embedding = result["embeddings"][0]

print("Original text:")
print(data["input"])

print("\nFirst 10 embedding numbers:")
print(embedding[:10])

print("\nTotal embedding numbers:", len(embedding))
