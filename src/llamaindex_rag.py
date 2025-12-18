"""
LlamaIndex RAG Pipeline
Similar structure to langchain_rag.py but using LlamaIndex framework
Compatible with both local and Google Colab environments
"""
import os
import sys
from pathlib import Path

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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

# -----------------------------
# 1. Load corpus
# -----------------------------
corpus_dir = BASE_DIR / "data" / "corpus" / "cuad"

# Check if index already exists
index_dir = BASE_DIR / "data" / "vectorstores" / "llamaindex_rag"
index_dir.mkdir(parents=True, exist_ok=True)

if (index_dir / "index.json").exists():
    print("Loading existing LlamaIndex vector store...")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    print("Vector store loaded")
else:
    print("Creating new LlamaIndex vector store...")
    
    # Load documents using SimpleDirectoryReader
    documents = SimpleDirectoryReader(str(corpus_dir)).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # -----------------------------
    # 2. Chunk documents (using node parser)
    # -----------------------------
    splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes (chunks)")
    
    # -----------------------------
    # 3. Create vector store index
    # -----------------------------
    # Use OpenAI embeddings
    embed_model = OpenAIEmbedding()
    
    # Create index from nodes
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
    )
    print("Vector store index created")
    
    # -----------------------------
    # 4. Save the index
    # -----------------------------
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"Index saved to {index_dir}")

# -----------------------------
# 5. Build query engine
# -----------------------------
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,  # Retrieve top 5 similar chunks
)

print("\nPipeline setup complete!")
print("Query engine ready. Use query_engine.query('your question') to query.")

