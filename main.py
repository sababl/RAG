import os
from typing import List, Optional, Tuple

import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv

from embedding import GeminiEmbeddingFunction, extract_text_from_pdf, chunk_text, BGEReranker

# Constants
PDF_FOLDER = "docs-pdf"
DB_NAME = "pydocs_db"

N_RESULTS = 3
MODEL_NAME = "gemini-2.0-flash"

doc_id_counter = 0 

def init_environment() -> None:
    """Initialize environment variables and API configuration."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)


def setup_database_from_folder(folder_path: str) -> chromadb.Collection:
    """
    Process all PDFs in a folder, chunk their content, and populate ChromaDB.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        chromadb.Collection: Configured database collection.
    """
    all_chunks = []
    all_ids = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {pdf_path}...")

            try:
                text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text)
                if chunks:
                    # Generate a unique ID for each chunk from this file
                    ids = [f"{filename}_{i}" for i in range(len(chunks))]
                    all_chunks.extend(chunks)
                    all_ids.extend(ids)

                print(f"Extracted {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return setup_database(all_chunks, all_ids)


def setup_database(content: List[str], ids: List[str]) -> chromadb.Collection:
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    client = chromadb.PersistentClient(path="./chroma_storage")

    db = client.get_or_create_collection(name="pydocs_db", embedding_function=embed_fn)

    if db.count() == 0:
        if len(content) != len(ids):
            raise ValueError(f"Document count ({len(content)}) does not match ID count ({len(ids)})")

        print(f"Adding {len(content)} documents in batches...")

        batch_size = 40000  # safely below ChromaDB limit
        for i in range(0, len(content), batch_size):
            chunk = content[i:i + batch_size]
            chunk_ids = ids[i:i + batch_size]
            db.add(documents=chunk, ids=chunk_ids)
            print(f"Added batch {i // batch_size + 1} ({len(chunk)} items)")
    
    return db

def generate_answer(question: str, context: List[str]) -> str:
    """
    Generate answer using Gemini model.

    Args:
        question (str): User's question
        context (List[str]): Retrieved context passages

    Returns:
        str: Generated answer
    """
    prompt = f"""
    You are an expert Python programming assistant.
    Answer the user's question clearly, accurately, and concisely based **only** on the provided context.
    If the context doesn't contain the necessary information, reply "I'm sorry, I couldn't find relevant information in the provided context."
    
    Format your response in two parts:
    1. First provide the answer without any citations
    2. Then add a "References:" section at the end that lists which passages (1-3) were used
    
    Example format:
    Answer: The explanation of the concept...

    References: [1, 2]

    Important:
    - Only use passage numbers 1-3
    - Don't include citations in the main answer text
    - List all used passages in the References section
    - If a passage wasn't used, don't list it in References

    Question: {question}

    Context: {context}
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def answer_question(user_question: str) -> tuple[str, str]:
    """
    Process user question and return answer using RAG.

    Args:
        user_question (str): User's Python-related question

    Returns:
        tuple[str, str]: Tuple containing (answer, passages)
    """
    try:
        embed_fn = GeminiEmbeddingFunction()
        reranker = BGEReranker()

        # Initial retrieval with embeddings
        embed_fn.document_mode = False
        results = db.query(
            query_texts=[user_question],
            n_results=N_RESULTS * 2  # Retrieve more passages for reranking
        )
        passages = results["documents"][0]

        # Rerank passages
        reranked_passages = reranker.rerank(user_question, passages, top_k=N_RESULTS)
        passages = [p[0] for p in reranked_passages]  # Extract just the passages, dropping scores
        
        passages_text = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return generate_answer(user_question, passages_text), passages_text

    except Exception as e:
        return f"Error processing question: {str(e)}", ""

# Initialize system
init_environment()
db = setup_database_from_folder(PDF_FOLDER)
