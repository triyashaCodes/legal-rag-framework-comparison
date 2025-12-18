import os
import json
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from benchmark_types import Benchmark

# Handle Colab vs local paths
if 'google.colab' in sys.modules:
    # In Colab, detect project directory
    current_dir = Path.cwd()
    if (current_dir / 'generate_cuad.py').exists():
        BASE_DIR = current_dir
    else:
        BASE_DIR = Path('/content')
        print("Warning: Using /content as BASE_DIR. Make sure you're in the project directory.")
    print("Running in Google Colab")
else:
    BASE_DIR = Path(__file__).parent
    print("Running locally")

# -----------------------------
# 1. Load corpus
# -----------------------------
corpus_dir = BASE_DIR / "data" / "corpus" / "cuad"
documents = []
for filename in os.listdir(corpus_dir):
    if filename.endswith(".txt"):
        filepath = corpus_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Create Document objects with metadata
            documents.append(Document(
                page_content=content,
                metadata={"source": filename}
            ))

print(f"Loaded {len(documents)} documents")

# -----------------------------
# 2. Chunk documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# -----------------------------
# 3. Create vector store (FAISS)
# -----------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created")

# -----------------------------
# 4. Build RetrievalQA chain
# -----------------------------
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

# -----------------------------
# 5. Save the pipeline for evaluation
# -----------------------------
# Save vectorstore and chain for later use
vectorstore_dir = BASE_DIR / "data" / "vectorstores"
vectorstore_dir.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(vectorstore_dir / "langchain_faiss"))
print(f"Vector store saved to {vectorstore_dir / 'langchain_faiss'}")

# Export the qa_chain (you'll reload it in evaluation script)
# Note: You'll need to recreate the chain in evaluation script with same config
print("\nPipeline setup complete!")

