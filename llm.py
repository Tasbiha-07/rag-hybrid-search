import os
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.documents import Document

load_dotenv()

def generate_answer(query: str, ranked_docs: List[Tuple[Document, float]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in .env file."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.8-flash")

    context_str = ""
    for idx, (doc, _) in enumerate(ranked_docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown File"))
        page = doc.metadata.get("page", 0) + 1
        context_str += f"\n[Passage {idx} | Source: {source} (Page {page})]:\n{doc.page_content}\n"

    prompt = f"""You are a precise technical question-answering assistant.
Answer the following question solely based on the provided context passages.
If the answer cannot be determined from the context, state clearly that the document doesn't contain that information.

Context:
{context_str}

Question:
{query}

Answer:"""

    response = model.generate_content(prompt)
    return response.text