"""
Web interface for the RAG (Retrieval-Augmented Generation) application.

This module provides a FastAPI web application that allows users to interact with
the RAG question-answering system through a simple web interface.
"""

from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import sys
import os

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "Default Model")


# Add parent directory to path to import from main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import answer_question

# Initialize FastAPI app
app = FastAPI(
    title="Python Documentation RAG Assistant",
    description="A retrieval-augmented generation system for Python documentation Q&A",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="webapp/templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """
    Render the main page with the question form.

    Args:
        request (Request): FastAPI request object

    Returns:
        HTMLResponse: Rendered template response
    """
    return templates.TemplateResponse(
        name="index.html",
        context={
            "request": request,
            "answer": None,
            "question": None,
            "passages": None,
            "model_name": MODEL_NAME
        }
    )


@app.post("/", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    question: str = Form(...)
) -> HTMLResponse:
    """
    Handle form submission and return the answer.

    Args:
        request (Request): FastAPI request object
        question (str): User's Python-related question

    Returns:
        HTMLResponse: Rendered template response with answer

    Raises:
        HTTPException: If question answering fails
    """
    try:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        answer, passages = answer_question(question)
        
        if not answer:
            raise ValueError("Could not generate an answer")

        # Escape any potential JavaScript in the content
        answer = str(answer).replace('`', '\\`')
        passages = str(passages).replace('`', '\\`')

        return templates.TemplateResponse(
            name="index.html",
            context={
                "request": request,
                "answer": answer,
                "question": question,
                "passages": passages,
                "model_name": MODEL_NAME
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )