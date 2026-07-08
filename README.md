# ChatPDF

A local Retrieval-Augmented Generation demo for chatting with uploaded PDFs.

## Stack

- Flask for the web app
- PyPDF for PDF text extraction
- Sentence Transformers for embeddings
- ChromaDB for local vector storage
- Ollama with Llama 3.2 1B for answer generation

## Setup

Use Python 3.11 or 3.12. Python 3.14 is not recommended for this project yet because some ML dependencies may not publish compatible wheels.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install Ollama from https://ollama.com, then pull the model:

```bash
ollama pull llama3.2:1b
```

The app uses `llama3.2:1b` by default because it runs on machines with less RAM. To use a larger model, set `OLLAMA_MODEL` before starting Flask.

## Run

Start Ollama in the background, then run:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Flow

1. Upload a PDF.
2. The app extracts readable text with PyPDF.
3. Text is split into overlapping chunks.
4. Chunks are embedded with `all-MiniLM-L6-v2`.
5. Embeddings and text chunks are stored in ChromaDB.
6. A question is embedded and used for similarity search.
7. Retrieved chunks are sent to Ollama with the question.
8. The answer is shown in the chat interface.

Each browser session gets a separate Chroma collection, which keeps one user's indexed PDF separate from another user's PDF.
