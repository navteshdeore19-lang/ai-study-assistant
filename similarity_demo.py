
import requests
import math

URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def get_embedding(text):
    response = requests.post(
        URL,
        json={"model": MODEL, "input": text},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


sentence_a = "Max pooling keeps the largest value in a window."
sentence_b = "Pooling selects the maximum number from a small region."
sentence_c = "A dog is playing in a garden."

embedding_a = get_embedding(sentence_a)
embedding_b = get_embedding(sentence_b)
embedding_c = get_embedding(sentence_c)

similarity_ab = cosine_similarity(embedding_a, embedding_b)
similarity_ac = cosine_similarity(embedding_a, embedding_c)

print("Sentence A:", sentence_a)
print("Sentence B:", sentence_b)
print("Sentence C:", sentence_c)

print(f"\nSimilarity between A and B: {similarity_ab:.4f}")
print(f"Similarity between A and C: {similarity_ac:.4f}")
