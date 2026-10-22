
# Simple RAG retrieval demonstration

chunks = [
    "A CNN is a deep learning model commonly used for image data.",
    "Max pooling keeps the largest value in each local window.",
    "ReLU replaces negative values with zero.",
    "Convolution filters detect patterns such as edges and textures."
]

question = input("Ask a question: ").lower()

# Find chunks containing words from the question
question_words = set(question.replace("?", "").split())

scores = []

for chunk in chunks:
    chunk_words = set(chunk.lower().replace(".", "").split())

    # Count matching words
    score = len(question_words & chunk_words)
    scores.append((score, chunk))

# Sort chunks by matching score, highest first
scores.sort(key=lambda item: item[0], reverse=True)

print("\nMost relevant chunk:")

if scores[0][0] > 0:
    print(scores[0][1])
    print("Matching words:", scores[0][0])
else:
    print("No relevant chunk found.")
