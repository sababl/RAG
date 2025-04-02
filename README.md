# Python RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers Python programming questions using content from "Think Python 2" book. The system uses Gemini embeddings and ChromaDB for efficient retrieval, with a clean web interface built using FastAPI.

## Features

- RAG-based question answering about Python programming
- Vector similarity search using ChromaDB
- Google's Gemini model for embeddings and text generation
- Web interface built with FastAPI and Tailwind CSS
- PDF text extraction and chunking capabilities

## Prerequisites

- Python 3.8+
- Google API key for Gemini
- Think Python 2 PDF file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG.git
cd RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install fastapi chromadb google-generativeai python-multipart python-dotenv pymupdf jinja2
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your Google API key.

5. Place the "Think Python 2" PDF file in the root directory as `thinkpython2.pdf`

## Usage

1. Start the web server:
```bash
uvicorn webapp.app:app --reload
```

2. Open your browser and navigate to `http://localhost:8000`

3. Enter your Python-related questions and get answers based on the book content

## Project Structure

```
RAG/
├── embedding.py      # Embedding and text processing functions
├── main.py          # Core RAG implementation
├── webapp/
│   ├── app.py       # FastAPI web application
│   └── templates/
│       └── index.html # Web interface template
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Think Python 2 by Allen B. Downey
- Google Gemini API
- ChromaDB team