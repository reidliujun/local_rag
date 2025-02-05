# Local RAG with Ollama

A local RAG (Retrieval Augmented Generation) implementation using Ollama for embeddings and LLM inference.

## Test

```bash
python -m unittest test_ingest.py -v
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pull required Ollama models:
```bash
# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model
ollama pull deepseek-r1
```

3. Configure document directory:
- Create a .env file in the project root
- Add your document directory path:
```bash
DOC_DIR=/path/to/your/markdown/docs/
```

## Usage
1. Start Ollama service:
```bash
ollama serve
```

2. Ingest documents and create embeddings:
```bash
python ingest.py
```

This will:

- Load markdown documents from your configured directory
- Split them into chunks
- Create embeddings using nomic-embed-text
- Store them in a local Chroma database

3. Launch the RAG interface:

```bash
python app.py
```

This will start a Gradio web interface where you can:

- Enter questions about your documents
- Get AI-generated answers based on the relevant context from your documents
The app uses:

- nomic-embed-text for document embeddings
- deepseek-r1 for generating answers
- ChromaDB for vector storage