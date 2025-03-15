from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import google.generativeai as genai
import fitz


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    chunk = ""
    for sentence in sentences:
        current_sentence = sentence + "."

        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            if chunk.strip(): 
                chunks.append(chunk.strip())
            words = chunk.split()
            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else ""
            chunk = overlap_text + " " + current_sentence + " "

    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks
