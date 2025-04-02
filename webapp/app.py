from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from main import answer_question

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, question: str = Form(...)):
    answer = answer_question(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "question": question})
