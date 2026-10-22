
from pypdf import PdfReader

pdf_path = "documents/notes.pdf"

# Step 1: Read the PDF
reader = PdfReader(pdf_path)

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

# Step 2: Set chunk size and overlap
chunk_size = 500
overlap = 100

chunks = []
start = 0

# Step 3: Create overlapping chunks
while start < len(pdf_text):

    end = min(start + chunk_size, len(pdf_text))

    # Try to end at a space instead of splitting a word
    if end < len(pdf_text):
        space_position = pdf_text.rfind(" ", start, end)

        if space_position > start:
            end = space_position

    chunk = pdf_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    if end == len(pdf_text):
        break

    # Start the next chunk slightly before the previous one ended
    next_start = max(start + 1, end - overlap)

    # Try to avoid starting in the middle of a word
    next_space = pdf_text.find(" ", next_start, end)

    if next_space != -1:
        next_start = next_space + 1

    start = next_start

# Step 4: Display the results
print("Total PDF pages:", len(reader.pages))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)
