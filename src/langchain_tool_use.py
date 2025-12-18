"""
LangChain RAG Pipeline with Tool-Use (New Agent API)
Uses 3 tools: date extraction, text summarization, and RAG retrieval
Compatible with both local and Google Colab environments
"""

import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents import create_agent


# ------------------------------------------------------
# Environment detection
# ------------------------------------------------------
if "google.colab" in sys.modules:
    current_dir = Path.cwd()
    BASE_DIR = current_dir if (current_dir / "generate_cuad.py").exists() else Path("/content")
    print("Running in Google Colab")
else:
    BASE_DIR = Path(__file__).parent
    print("Running locally")

# GLOBAL VECTORSTORE
_vectorstore = None


# ------------------------------------------------------
# Tool 1: Extract Dates
# ------------------------------------------------------

def extract_dates(text: str) -> str:
    """Extracts all date expressions from the given legal text.
    
    Use this tool when:
    - The query asks about dates, expiration dates, renewal terms, deadlines, or time periods
    - You have contract text and need to find specific dates mentioned in it
    - Examples: "What is the expiration date?", "When does this contract expire?", "What is the renewal term?"
    
    Input: The text from which to extract dates (can be contract text or a clause).
    Output: A comma-separated list of all dates found in the text.
    """
    import re

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+(day|days|month|months|year|years)\b",
    ]

    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)

    seen = set()
    cleaned = []
    for d in dates:
        d = d if isinstance(d, str) else " ".join(d) if isinstance(d, tuple) else str(d)
        if d not in seen:
            cleaned.append(d)
            seen.add(d)

    return ", ".join(cleaned) if cleaned else "No dates found."


# ------------------------------------------------------
# Tool 2: Summarize Text
# ------------------------------------------------------
@tool
def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Produces a concise summary of the provided legal text."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = f"""
    Summarize the following contract text in {max_sentences} sentences.

    Text:
    {text[:2000]}

    Summary:
    """

    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error summarizing: {str(e)}"


# ------------------------------------------------------
# Tool 3: RAG Retrieval
# ------------------------------------------------------
@tool
def rag_retrieve(query: str) -> str:
    """Retrieves relevant legal documents or clauses for the given query."""
    global _vectorstore

    if _vectorstore is None:
        raise ValueError("Vectorstore not initialized. Call setup_pipeline first.")

    docs = _vectorstore.similarity_search(query, k=10)
    if not docs:
        return f"No relevant information found for: {query}"

    return "\n\n".join([doc.page_content for doc in docs[:3]])[:2000]


# ------------------------------------------------------
# Setup Pipeline (Vectorstore + Agent)
# ------------------------------------------------------
def setup_pipeline():
    global _vectorstore

    vectorstore_dir = BASE_DIR / "data" / "vectorstores"
    vectorstore_path = vectorstore_dir / "langchain_faiss"

    embeddings = OpenAIEmbeddings()

    # Load or build FAISS index
    if (vectorstore_path / "index.faiss").exists():
        print("Loading existing vectorstore...")
        _vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Building new vectorstore...")
        corpus_dir = BASE_DIR / "data" / "corpus" / "cuad"

        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".txt"):
                with open(corpus_dir / filename, "r", encoding="utf-8") as f:
                    documents.append(
                        Document(page_content=f.read(), metadata={"source": filename})
                    )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        _vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        _vectorstore.save_local(str(vectorstore_path))
        print("Vectorstore saved.")

    # Tool list
    tools = [extract_dates, summarize_text, rag_retrieve]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # New agent API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a legal analysis assistant who uses tools when helpful.")


    return agent


if __name__ == "__main__":
    agent = setup_pipeline()
    print("\nPipeline setup complete!")
    print("Run: python evaluate_langchain_tool_use.py")
