
text = """
CNNs are used for image processing.
Pooling reduces feature-map size.
ReLU replaces negative values with zero.
"""

# Split the text into sentences
chunks = [
    sentence.strip()
    for sentence in text.strip().split(".")
    if sentence.strip()
]

# Display each chunk
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}.")
