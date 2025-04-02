import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb

from embedding import GeminiEmbeddingFunction, extract_text_from_pdf, chunk_text

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

content = extract_text_from_pdf("thinkpython2.pdf")
chunks = chunk_text(content)

DB_NAME = "pythonDocDB"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

print("db count", db.count())

#--------------------------------retriever-------------------------------
def answer_question(user_question: str) -> str:
    try:
        embed_fn.document_mode = False
        results = db.query(query_texts=[user_question], n_results=3)
        passages = results["documents"][0]
    except Exception as e:
        return f"Error during query: {str(e)}"

    prompt = f"""
    You are an expert Python programming assistant.
    Answer the user's question clearly, accurately, and concisely based **only** on the provided context.
    If the context doesn't contain the necessary information, reply "I'm sorry, I couldn't find relevant information in the provided context."

    Question: {user_question}

    Context: {passages}
    """

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text