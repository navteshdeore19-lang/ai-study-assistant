# AI Study Assistant

A local AI-powered study assistant that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- Ask general study questions using a local language model.
- Upload multiple PDF documents.
- Split PDF text into smaller chunks.
- Generate embeddings using Nomic Embed Text through Ollama.
- Search for relevant passages using ChromaDB.
- Generate answers using Qwen 2.5 3B.
- Display source PDF names alongside answers.
- Store document embeddings in a local persistent vector database.
- Use a Streamlit web interface.

## Architecture

1. **PDF Processing:** Extract text from uploaded PDF documents using pypdf.
2. **Chunking:** Split extracted text into smaller overlapping sections.
3. **Embeddings:** Convert text into numerical vectors using Nomic Embed Text.
4. **Vector Database:** Store and retrieve document vectors using ChromaDB.
5. **Retrieval:** Find passages relevant to the user's question.
6. **Generation:** Use Qwen 2.5 3B through Ollama to generate an answer from the retrieved context.
7. **Interface:** Provide an interactive browser-based experience with Streamlit.

## Technology Stack

- Python
- Streamlit
- Ollama
- Qwen 2.5 3B
- Nomic Embed Text
- ChromaDB
- PyPDF

## Prerequisites

Install Python and Ollama. The following Ollama models are used:

- `qwen2.5:3b`
- `nomic-embed-text`

Download the models using:

```powershell

ollama pull qwen2.5:3b
ollama pull nomic-embed-text

Make sure Ollama is running before launching the application.

## Installation

Clone or download this repository and open a terminal in the project folder.

Install the required Python packages:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run web_app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

Choose General Chat to ask general questions, or Ask My PDFs to upload documents and ask questions about their contents.

## Local Data and Privacy

The application uses locally running Ollama models. Uploaded PDFs and their embeddings are processed locally by the application.

The ChromaDB database is stored in the local `chroma_db` directory. It is excluded from Git by `.gitignore` and is not included in the repository by default.

Do not upload private documents, credentials, API keys, or other confidential information to a public repository.

## Current Limitations

- PDF extraction works best with selectable text; scanned documents may require OCR.
- Answer quality depends on the uploaded document content and retrieved passages.
- Large PDFs may take time to process and embed.
- Ollama and the required local models must be available for answers to be generated.

## Future Improvements

- Improve document replacement and deletion handling.
- Add automated tests and stronger error handling.
- Improve the user interface and chat experience.
- Add document management and retrieval controls.
- Prepare a deployment-friendly version with clear configuration and security guidance.

## Project Status

The multiple-PDF Streamlit application has been integrated with Ollama and ChromaDB and manually tested. Additional testing and documentation improvements are ongoing.
