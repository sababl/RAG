# Retrieval-Augmented Python QA System

This repository implements a Retrieval-Augmented Generation (RAG) system specialized for answering Python programming questions. It uses ChromaDB for dense passage retrieval, Google Generative AI for embeddings and generation, and a cross-encoder reranker to refine results. A simple FastAPI web interface serves the QA functionality.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Build/Populate the Vector Database](#1-buildpopulate-the-vector-database)
  - [2. Run the QA API Server](#2-run-the-qa-api-server)
  - [3. Evaluate System Performance](#3-evaluate-system-performance)
- [Customization](#customization)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Features

- **PDF Ingestion & Chunking**: Extract text from Python documentation PDFs and split into word-based chunks with adjustable overlap.
- **Dense Retrieval**: Use Google’s `text-embedding-004` model to index and retrieve relevant passages via ChromaDB.
- **Reranking**: Improve retrieval precision with a BGE-based cross-encoder reranker.
- **Answer Generation**: Generate answers conditioned on retrieved context using Google's `gemini-2.0-flash` model.
- **Web Interface**: FastAPI application (`app.py`) for interactive Q&A.
- **Evaluation**: Automated scripts to compute BLEU, ROUGE‑L, METEOR, BERTScore, and Cosine Similarity on a held-out dataset.

---

## Repository Structure

```
├── docs-pdf/               # Folder containing Python documentation PDFs
├── embedding.py            # PDF parsing, chunking, and embedding utilities
├── main.py                 # Core RAG pipeline: retrieval, reranking, and generation
├── app.py                  # FastAPI web server for QA interface
├── evaluate.py             # Scripts for generating and scoring model outputs
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── README.md               # This documentation
```

---

## Prerequisites

- Python 3.9+
- Google Cloud API key with access to the Generative AI API
- A local or remote GPU for BGE reranking (optional but recommended)

---

## Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/python-rag-qa.git
   cd python-rag-qa
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   .\\venv\\Scripts\\activate  # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download Python documentation PDFs**
   - Place all `.pdf` files (e.g., Python 3.13 docs) into the `docs-pdf/` directory.

---

## Configuration

1. **Environment variables**
   Copy `.env.example` to `.env` and set your Google API key:
   ```ini
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. **Database location**
   - By default, ChromaDB will persist data under `./chroma_storage`. You can modify the path in `main.py` if needed.

---

## Usage

### 1. Build/Populate the Vector Database

This step processes all PDFs in `docs-pdf/`, chunks the text, generates embeddings, and stores them in ChromaDB.

```bash
python main.py
```

You should see console output indicating how many chunks were extracted and added.

### 2. Run the QA API Server

Start the FastAPI server to serve the QA interface:

```bash
uvicorn app:app --reload
```

- Open your browser at `http://127.0.0.1:8000`.
- Submit Python questions via the web form and receive answers based only on retrieved context.

### 3. Evaluate System Performance

1. **Generate evaluation CSV** (if you need to regenerate generated answers):
   ```bash
   # Uncomment and adjust paths in evaluate.py, then run:
   python evaluate.py --generate
   ```

2. **Compute metrics** against a reference dataset (`evaluation_results.csv`):
   ```bash
   python evaluate.py
   ```

- Results are saved to `final_evaluation_scores.csv` and `average_scores.csv`.

---

## Customization

- **Chunking parameters**: In `embedding.py`, adjust `chunk_size` (words per chunk) and `overlap` (words of context carryover).
- **Retrieval settings**: Change `N_RESULTS` in `main.py` to retrieve more or fewer passages.
- **Generation model**: Switch `MODEL_NAME` to experiment with different LLM backends (e.g., `gemini-1.5-flash`).
- **Reranker**: To disable reranking, comment out the reranking block in `answer_question`.

---

## Future Improvements

- Implement RAG-Token variant for token-level document conditioning.
- Integrate live sources (GitHub, StackOverflow) for dynamic knowledge updates.
- Add interactive UI features to trace which passages were used in each answer.
- Extend evaluation with Mean Reciprocal Rank (MRR) for retrieval quality.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
