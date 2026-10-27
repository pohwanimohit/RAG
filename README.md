# Multimodal RAG Ingestion Pipeline

A Python-based document ingestion pipeline for a Retrieval-Augmented Generation (RAG) system. 

## Features

* **PDF ingestion:** Extract text and page metadata from PDF files.
* **Video transcription:** Transcribe audio using Faster-Whisper.
* **Text chunking:** Split extracted content into overlapping chunks using LangChain.
* **Embeddings:** Generate semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector storage:** Store documents, metadata, and embeddings in ChromaDB.
* **Semantic search:** Retrieve relevant chunks using vector similarity.

## Project Structure

```text
RAG/
├── README.md
├── .gitignore
└── ingestion_pipeline/
    ├── main.py
    ├── pdf_processor.py
    ├── video_processor.py
    ├── document_processor.py
    ├── vector_store.py
```

Generated folders such as `__pycache__/` and `chroma_db/` are excluded from version control.

## Tech Stack

* Python
* LangChain
* PyPDFLoader
* Faster-Whisper
* Hugging Face Sentence Transformers
* ChromaDB

## Setup

### Dependencies

Install the packages required by the project:

```bash
pip install langchain-community langchain-text-splitters langchain-huggingface langchain-chroma faster-whisper sentence-transformers pypdf
```




Add a license before distributing or reusing this project. If the project uses third-party assets, ensure you have permission to redistribute them.
