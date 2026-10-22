
import requests
import math
from pypdf import PdfReader

# Ollama embedding API
URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


# Convert text into an embedding
def get_embedding(text):
    response = requests.post(
        URL,
        json={"model": MODEL, "input": text},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


# Compare two embeddings
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Read PDF text
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# Split text into overlapping chunks
def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            space_position = text.rfind(" ", start, end)
            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        next_start = max(start + 1, end - overlap)
        next_space = text.find(" ", next_start, end)

        if next_space != -1:
            next_start = next_space + 1

        start = next_start

    return chunks


# Main program
pdf_text = read_pdf("documents/notes.pdf")
chunks = create_chunks(pdf_text)

print("PDF loaded successfully.")
print("Total chunks:", len(chunks))
print("\nCreating embeddings for PDF chunks...")

chunk_embeddings = []

for i, chunk in enumerate(chunks, start=1):
    embedding = get_embedding(chunk)
    chunk_embeddings.append(embedding)
    print(f"Processed chunk {i}/{len(chunks)}")

print("\nSemantic search is ready!")

question = input("\nAsk a question about your PDF: ").strip()

if not question:
    print("Please enter a question.")
    raise SystemExit

question_embedding = get_embedding(question)

# Score each chunk against the question
results = []

for chunk, embedding in zip(chunks, chunk_embeddings):
    score = cosine_similarity(question_embedding, embedding)
    results.append((score, chunk))

# Show the most relevant chunks first
results.sort(key=lambda item: item[0], reverse=True)

print("\nMost relevant chunks:")

for rank, (score, chunk) in enumerate(results[:3], start=1):
    print(f"\n--- Result {rank} | Similarity: {score:.4f} ---")
    print(chunk)
