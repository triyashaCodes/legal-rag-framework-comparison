import os
import json
import time
import sys
from typing import List, Dict
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from benchmark_types import Benchmark, QAGroundTruth

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Metrics will not be logged.")

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
# Evaluation Metrics
# -----------------------------


def f1_score(predicted: str, gold: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def retrieval_recall(retrieved_chunks: List[str], gold_snippets: List[Dict], k: int = 5) -> float:
    """Calculate retrieval recall@k - how many gold snippets were retrieved."""
    if not gold_snippets:
        return 0.0
    
    # Get gold text content
    gold_texts = []
    corpus_base = BASE_DIR / "data" / "corpus"
    for snippet in gold_snippets:
        filepath = corpus_base / snippet["file_path"]
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                gold_text = text[snippet["span"][0]:snippet["span"][1]]
                gold_texts.append(gold_text.lower())
    
    if not gold_texts:
        return 0.0
    
    # Check if any gold text appears in retrieved chunks
    retrieved_text = " ".join([chunk.lower() for chunk in retrieved_chunks[:k]])
    
    found_count = 0
    for gold_text in gold_texts:
        # Check if gold text (or significant portion) appears in retrieved chunks
        if gold_text[:100] in retrieved_text or any(gold_text in chunk.lower() for chunk in retrieved_chunks[:k]):
            found_count += 1
    
    return found_count / len(gold_texts)


# -----------------------------
# Load RAG Pipeline
# -----------------------------

def load_langchain_pipeline():
    """Load the pre-built LangChain RAG pipeline."""
    embeddings = OpenAIEmbeddings()
    vectorstore_path = BASE_DIR / "data" / "vectorstores" / "langchain_faiss"
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    
    return qa_chain


# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate_on_benchmark(qa_chain, benchmark: Benchmark, max_cases: int = None) -> Dict:
    """Evaluate RAG pipeline on benchmark."""
    test_cases = benchmark.tests
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    results = {
        "f1_scores": [],
        "retrieval_recalls": [],
        "latencies": [],
        "detailed_results": []
    }
    
    print(f"Evaluating on {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases):
        query = test_case.query
        
        # Time the query
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        latency = time.time() - start_time
        
        answer = result["result"]
        retrieved_docs = [doc.page_content for doc in result.get("source_documents", [])]
        
        # Get gold answers
        gold_texts = []
        for snippet in test_case.snippets:
            filepath = os.path.join("./data/corpus", snippet.file_path)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    gold_text = text[snippet.span[0]:snippet.span[1]]
                    gold_texts.append(gold_text)
        
        # Calculate metrics
        # For multiple gold snippets, use the best match
        best_f1 = 0.0
        for gold_text in gold_texts:
            f1 = f1_score(answer, gold_text)
            best_f1 = max(best_f1, f1)
        
        # Retrieval recall
        # Convert Pydantic objects to dicts for retrieval_recall function
        snippet_dicts = [
            {
                "file_path": s.file_path,
                "span": list(s.span) if isinstance(s.span, tuple) else s.span
            }
            for s in test_case.snippets
        ]
        recall = retrieval_recall(retrieved_docs, snippet_dicts, k=5)
        
        results["f1_scores"].append(best_f1)
        results["retrieval_recalls"].append(recall)
        results["latencies"].append(latency)
        
        results["detailed_results"].append({
            "query": query,
            "answer": answer,
            "gold_answers": gold_texts,
            "f1": best_f1,
            "retrieval_recall": recall,
            "latency": latency
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_cases)} cases...")
    
    return results


# -----------------------------
# Main Evaluation
# -----------------------------

def main():
    # Initialize wandb (if available)
    if WANDB_AVAILABLE:
        wandb.init(
            project="legal-rag-evaluation",
            name="langchain-rag-cuad",
            config={
                "framework": "LangChain",
                "dataset": "CUAD",
                "model": "gpt-3.5-turbo",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "retrieval_k": 5,
            }
        )
    
    # Load benchmark
    print("Loading benchmark...")
    benchmark_path = BASE_DIR / "data" / "benchmarks" / "cuad.json"
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)
    
    benchmark = Benchmark(**benchmark_data)
    print(f"Loaded benchmark with {len(benchmark.tests)} test cases")
    
    # Load pipeline
    print("\nLoading LangChain RAG pipeline...")
    qa_chain = load_langchain_pipeline()
    print("Pipeline loaded")
    
    # Evaluate (start with 50 cases for testing, remove limit for full evaluation)
    print("\nStarting evaluation...")
    results = evaluate_on_benchmark(qa_chain, benchmark, max_cases=50)
    
    # Calculate summary statistics
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
    avg_recall = sum(results["retrieval_recalls"]) / len(results["retrieval_recalls"])
    avg_latency = sum(results["latencies"]) / len(results["latencies"])
    
    # Note: exact_match removed from tool-use, but kept here for RAG comparison
    # avg_em = sum(results.get("exact_matches", [0.0])) / len(results.get("exact_matches", [0.0])) if results.get("exact_matches") else 0.0
    
    # Log metrics to wandb (if available)
    if WANDB_AVAILABLE:
        wandb.log({
            "f1_score": avg_f1,
            "retrieval_recall@5": avg_recall,
            "avg_latency": avg_latency,
            "num_test_cases": len(results["f1_scores"]),
        })
        
        # Log per-case metrics as a table
        table_data = []
        for i, detail in enumerate(results["detailed_results"][:10]):  # Log first 10 examples
            table_data.append([
                i + 1,
                detail["query"][:100] + "...",  # Truncate long queries
                detail["answer"][:100] + "...",  # Truncate long answers
                detail["f1"],
                detail["retrieval_recall"],
                detail["latency"],
            ])
        
        wandb.log({
            "examples": wandb.Table(
                columns=["Case", "Query", "Answer", "F1", "Recall@5", "Latency"],
                data=table_data
            )
        })
        
        # Log distribution of scores
        wandb.log({
            "f1_distribution": wandb.Histogram(results["f1_scores"]),
            "recall_distribution": wandb.Histogram(results["retrieval_recalls"]),
            "latency_distribution": wandb.Histogram(results["latencies"]),
        })
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"F1 Score: {avg_f1:.2%}")
    print(f"Retrieval Recall@5: {avg_recall:.2%}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print("="*50)
    
    if WANDB_AVAILABLE:
        print(f"\nResults logged to wandb: {wandb.run.url}")
        wandb_url = wandb.run.url
    else:
        wandb_url = None
    
    # Save detailed results
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "langchain_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "num_cases": len(results["f1_scores"]),
                "f1_score": avg_f1,
                "retrieval_recall": avg_recall,
                "avg_latency": avg_latency,
                "wandb_run_url": wandb_url,
            },
            "detailed_results": results["detailed_results"]
        }, f, indent=2)
    
    print(f"Detailed results saved to {output_path}")
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

