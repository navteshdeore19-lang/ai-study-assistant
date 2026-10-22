# AI Study Assistant

A local AI-powered study assistant that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- Ask general study questions using a local language model.
- Upload multiple PDF documents.
- Split PDF text into smaller overlapping chunks.
- Generate embeddings using Nomic Embed Text through Ollama.
- Retrieve relevant passages using ChromaDB.
- Generate answers using Qwen 2.5 3B.
- Display source PDF names alongside answers.
- Store document embeddings in a local persistent vector database.
- Use an interactive Streamlit web interface.

## Architecture

1. **PDF Processing:** Extract text from PDF documents using pypdf.
2. **Chunking:** Split extracted text into smaller overlapping sections.
3. **Embeddings:** Convert text into numerical vectors using Nomic Embed Text.
4. **Vector Database:** Store and retrieve document vectors using ChromaDB.
5. **Retrieval:** Find passages relevant to the user's question.
6. **Generation:** Use Qwen 2.5 3B through Ollama to generate answers from retrieved context.
7. **Interface:** Provide a browser-based experience with Streamlit.

## Technology Stack

- Python
- Streamlit
- Ollama
- Qwen 2.5 3B
- Nomic Embed Text
- ChromaDB
- pypdf

## Prerequisites

Install Python and [Ollama](https://ollama.com/).

Download the required models in PowerShell:

```powershell
ollama pull qwen2.5:3b
```

```powershell
ollama pull nomic-embed-text
```

Make sure Ollama is running before launching the application.

## Installation

Clone this repository:

```powershell
git clone https://github.com/navteshdeore19-lang/ai-study-assistant.git
```

Open the project folder:

```powershell
cd ai-study-assistant
```

Install the required Python packages:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run web_app.py
```

Open the local URL shown in your terminal, usually:

http://localhost:8501

Choose **General Chat** to ask general questions, or **Ask My PDFs** to upload documents and ask questions about their contents.

## Local Data and Privacy

The application uses locally running Ollama models. PDF processing, embedding generation, and answer generation are designed to run locally.

The ChromaDB database is stored in the local `chroma_db` directory. This directory is excluded from Git.

Do not upload private documents, credentials, API keys, or confidential information to a public repository.

## Current Limitations

- PDF extraction works best with selectable text; scanned documents may require OCR.
- Answer quality depends on the uploaded documents and retrieved passages.
- Large PDFs may take time to process and embed.
- Ollama and the required models must be available for answers to be generated.

## Future Improvements

- Add automated tests and stronger error handling.
- Improve document replacement and deletion handling.
- Add document management and retrieval controls.
- Improve the user interface and chat experience.
- Explore deployment options with appropriate configuration and security guidance.

## Project Status

The multiple-PDF Streamlit application has been manually tested for general chat, multiple-PDF question answering, and source-name display. Further testing and improvements are ongoing.