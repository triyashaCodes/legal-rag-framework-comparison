#!/usr/bin/env python3
"""
Evaluate LlamaIndex RAG on CUAD benchmark.

Evaluator for LlamaIndex RAG pipeline.
Tests RAG's ability to retrieve and answer questions from legal documents.
Saves detailed JSON results. Compatible with query_engine.query() style.
"""

import os
import json
import time
import sys
from typing import List, Dict, Any
from pathlib import Path

# -------------------------------
# ENV / PATH
# -------------------------------
if 'google.colab' in sys.modules:
    current_dir = Path.cwd()
    BASE_DIR = current_dir if (current_dir / 'generate_cuad.py').exists() else Path('/content')
else:
    BASE_DIR = Path(__file__).parent

# -------------------------------
# IMPORTS
# -------------------------------
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from benchmark_types import Benchmark

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# -------------------------------
# METRICS
# -------------------------------
def f1_score(predicted: str, gold: str) -> float:
    """
    Calculate token-level F1 score between predicted and gold answers.
    
    Computes precision and recall based on token overlap, then calculates
    F1 score as the harmonic mean of precision and recall.
    
    Args:
        predicted: The predicted answer string.
        gold: The gold/ground truth answer string.
        
    Returns:
        F1 score between 0.0 and 1.0 (higher is better).
    """
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def retrieval_recall(retrieved_chunks: List[str], gold_snippets: List[Dict], k: int = 5) -> float:
    """
    Calculate retrieval recall@k - how many gold snippets were retrieved.
    
    Args:
        retrieved_chunks: List of retrieved chunk texts.
        gold_snippets: List of gold snippet dictionaries with file_path and span.
        k: Number of top chunks to consider (default: 5).
        
    Returns:
        Recall score between 0.0 and 1.0.
    """
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

# -------------------------------
# Load RAG Pipeline
# -------------------------------
def load_llamaindex_pipeline():
    """
    Load the pre-built LlamaIndex RAG pipeline.
    
    Returns:
        LlamaIndex query engine ready for queries.
    """
    index_dir = BASE_DIR / "data" / "vectorstores" / "llamaindex_rag"
    
    if not (index_dir / "index.json").exists():
        raise FileNotFoundError(
            f"LlamaIndex index not found at {index_dir}. "
            "Please run llamaindex_rag.py first to create the index."
        )
    
    print(f"Loading LlamaIndex vector store from {index_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)
    
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,  # Retrieve top 5 similar chunks
    )
    
    print("LlamaIndex RAG pipeline loaded")
    return query_engine

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_on_benchmark(query_engine, benchmark: Benchmark, max_cases: int = None, recall_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline on benchmark.
    
    Runs the query engine on each test case, extracts answers and retrieved chunks,
    calculates metrics (F1, retrieval recall@k), and saves results to JSON.
    
    Args:
        query_engine: LlamaIndex query engine to evaluate.
        benchmark: Benchmark object containing test cases with queries and snippets.
        max_cases: Maximum number of test cases to evaluate (None for all).
        recall_k: K value for recall@k calculation (default: 5).
        
    Returns:
        Dictionary containing:
        - f1_scores: List of F1 scores per test case
        - retrieval_recalls: List of recall@k scores
        - latencies: List of latencies per test case
        - detailed_results: List of detailed per-case results
    """
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
        print(f"[{i+1}/{len(test_cases)}] Query: {query[:80]}...")

        start = time.time()
        try:
            # Real query - always use actual LLM calls
            response = query_engine.query(query)
            latency = time.time() - start
            
            answer = str(response.response) if hasattr(response, 'response') else str(response)
            
            # Extract retrieved chunks from source nodes
            retrieved_docs = []
            if hasattr(response, 'source_nodes'):
                retrieved_docs = [node.text for node in response.source_nodes[:recall_k]]
            elif hasattr(response, 'get_formatted_sources'):
                retrieved_docs = [response.get_formatted_sources()]

            # Get gold answers
            gold_texts = []
            corpus_dir = BASE_DIR / "data" / "corpus"
            for snippet in test_case.snippets:
                filepath = corpus_dir / snippet.file_path
                if filepath.exists():
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
            snippet_dicts = [
                {
                    "file_path": s.file_path,
                    "span": list(s.span) if isinstance(s.span, tuple) else s.span
                }
                for s in test_case.snippets
            ]
            recall = retrieval_recall(retrieved_docs, snippet_dicts, k=recall_k)

            results["f1_scores"].append(best_f1)
            results["retrieval_recalls"].append(recall)
            results["latencies"].append(latency)

            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "gold_answers": gold_texts,
                "f1": best_f1,
                "retrieval_recall": recall,
                "latency": latency,
                "retrieved_chunks_count": len(retrieved_docs)
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_cases)} cases...")

        except Exception as e:
            latency = time.time() - start
            print(f"  ERROR: {e}")
            
            best_f1 = 0.0
            recall = 0.0
            answer = f"ERROR: {str(e)}"
            retrieved_docs = []
            
            results["f1_scores"].append(best_f1)
            results["retrieval_recalls"].append(recall)
            results["latencies"].append(latency)
            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "gold_answers": [],
                "f1": best_f1,
                "retrieval_recall": recall,
                "latency": latency,
                "retrieved_chunks_count": len(retrieved_docs),
                "error": str(e)
            })

    return results

# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main evaluation function.
    
    Loads CUAD benchmark, initializes LlamaIndex RAG pipeline,
    runs evaluation, logs to wandb, and prints summary statistics.
    """
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="legal-rag-evaluation",
            name="llamaindex-rag-cuad",
            config={
                "framework": "LlamaIndex",
                "mode": "rag",
                "dataset": "CUAD",
                "model": "gpt-3.5-turbo",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "retrieval_k": 5,
            }
        )
    
    total_start_time = time.time()
    
    # Load benchmark
    print("Loading benchmark...")
    benchmark_path = BASE_DIR / "data" / "benchmarks" / "cuad.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    
    benchmark = Benchmark(**benchmark_data)
    print(f"Loaded benchmark with {len(benchmark.tests)} test cases")

    # Load pipeline
    print("\nLoading LlamaIndex RAG pipeline...")
    query_engine = load_llamaindex_pipeline()

    # Evaluate (start with 50 cases for testing, remove limit for full evaluation)
    print("\nStarting evaluation...")
    results = evaluate_on_benchmark(query_engine, benchmark, max_cases=50, recall_k=5)
    
    total_time = time.time() - total_start_time

    # Calculate summary statistics
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"]) if results["f1_scores"] else 0.0
    avg_recall = sum(results["retrieval_recalls"]) / len(results["retrieval_recalls"]) if results["retrieval_recalls"] else 0.0
    avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0.0

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
                detail["query"][:100] + "..." if len(detail["query"]) > 100 else detail["query"],
                detail["answer"][:100] + "..." if len(detail.get("answer", "")) > 100 else detail.get("answer", ""),
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
        
        wandb.log({"total_evaluation_time": total_time})

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"F1 Score: {avg_f1:.2%}")
    print(f"Retrieval Recall@5: {avg_recall:.2%}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Total Evaluation Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*50)

    if WANDB_AVAILABLE:
        print(f"\nResults logged to wandb: {wandb.run.url}")
        wandb_url = wandb.run.url
    else:
        wandb_url = None

    # Save detailed results
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "llamaindex_rag_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "num_cases": len(results["f1_scores"]),
                "f1_score": avg_f1,
                "retrieval_recall": avg_recall,
                "avg_latency": avg_latency,
                "total_evaluation_time": total_time,
                "wandb_run_url": wandb_url,
            },
            "detailed_results": results["detailed_results"]
        }, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to {output_path}")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

