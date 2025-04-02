from typing import Optional

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from main import answer_question


app = FastAPI()
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
            "question": None
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

        answer = answer_question(question)
        
        if not answer:
            raise ValueError("Could not generate an answer")

        return templates.TemplateResponse(
            name="index.html",
            context={
                "request": request,
                "answer": answer,
                "question": question
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)