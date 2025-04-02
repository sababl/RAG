"""
Main module for Python RAG system using Gemini and ChromaDB.
Handles document processing, embedding, and question answering.
"""

import os
from typing import List, Optional

import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

from embedding import GeminiEmbeddingFunction, extract_text_from_pdf, chunk_text

# Constants
DB_NAME = "pythonDocDB"
MODEL_NAME = "gemini-1.5-flash-latest"
PDF_PATH = "thinkpython2.pdf"
N_RESULTS = 3

def init_environment() -> None:
    """Initialize environment variables and API configuration."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)

def setup_database(content: List[str]) -> chromadb.Collection:
    """
    Set up and populate ChromaDB with document chunks.

    Args:
        content (List[str]): List of text chunks to store

    Returns:
        chromadb.Collection: Configured database collection
    """
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    client = chromadb.Client()
    db = client.get_or_create_collection(
        name=DB_NAME,
        embedding_function=embed_fn
    )

    # Add documents if collection is empty
    if db.count() == 0:
        db.add(
            documents=content,
            ids=[str(i) for i in range(len(content))]
        )
        print(f"Added {db.count()} documents to database")
    
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

    Question: {question}

    Context: {context}
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

def answer_question(user_question: str) -> str:
    """
    Process user question and return answer using RAG.

    Args:
        user_question (str): User's Python-related question

    Returns:
        str: Answer to the question
    """
    try:
        embed_fn = GeminiEmbeddingFunction()

        embed_fn.document_mode = False
        results = db.query(
            query_texts=[user_question],
            n_results=N_RESULTS
        )
        passages = results["documents"][0]
        
        return generate_answer(user_question, passages)

    except Exception as e:
        return f"Error processing question: {str(e)}"

# Initialize system
try:
    init_environment()
    content = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(content)
    db = setup_database(chunks)

except Exception as e:
    print(f"Error initializing system: {str(e)}")
    raise