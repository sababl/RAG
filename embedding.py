from typing import List
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import google.generativeai as genai
import fitz


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function using Google Generative AI to generate embeddings for documents or queries.

    Attributes:
        document_mode (bool): If True, embeddings are generated for documents; otherwise, for queries.
    """

    def __init__(self, document_mode: bool = True) -> None:
        self.document_mode = document_mode

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given input documents or queries.

        Args:
            input (Documents): A list of document strings.

        Returns:
            Embeddings: The generated embeddings.
        """

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


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The concatenated text from all pages.
    """
    pages_text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages_text.append(page.get_text())
    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits the input text into chunks of a specified maximum size with a given overlap.

    This implementation splits the text into sentences based on periods. It then accumulates sentences
    into chunks until the chunk size is reached, then starts a new chunk including a specified number
    of overlapping words from the previous chunk.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum character length of each chunk.
        overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence_with_period = sentence + "."
        if len(current_chunk) + len(sentence_with_period) <= chunk_size:
            current_chunk += sentence_with_period + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Compute overlap from the current chunk
            words = current_chunk.split()
            overlap_text = (
                " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
            )
            current_chunk = overlap_text + " " + sentence_with_period + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
