
from pypdf import PdfReader

pdf_path = "documents/notes.pdf"

reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))
print("\nPDF TEXT:\n")

for page in reader.pages:
    text = page.extract_text()
    if text:
        print(text)
