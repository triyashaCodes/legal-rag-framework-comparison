#!/usr/bin/env python3
"""
Evaluate LlamaIndex Tool-Use (Agent) on CUAD benchmark.

Uses the latest LlamaIndex agent API with async invocation
"""

import os
import json
import time
import asyncio
import nest_asyncio
import re
import csv
from pathlib import Path
from typing import List, Dict, Any
import sys

# Allow nested loops (Colab / Jupyter)
nest_asyncio.apply()

# -------------------------------
# ENV / PATH
# -------------------------------
if "google.colab" in sys.modules:
    BASE_DIR = Path("/content/CUAD-RAG")
else:
    BASE_DIR = Path(__file__).parent

# -------------------------------
# IMPORTS (LATEST)
# -------------------------------
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from benchmark_types import Benchmark

# -------------------------------
# WANDB
# -------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------------------
# METRICS
# -------------------------------
def f1_score(predicted: str, gold: str) -> float:
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & gold_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def extract_final_answer(response) -> str:
    """
    Extract the final answer from LlamaIndex agent's response.
    
    Args:
        response: LlamaIndex agent response object.
        
    Returns:
        The final answer string, or empty string if no answer found.
    """
    if response is None:
        return ""
    if hasattr(response, "response"):
        return str(response.response).strip()
    elif hasattr(response, "message"):
        if hasattr(response.message, "content"):
            return str(response.message.content).strip()
        return str(response.message).strip()
    elif isinstance(response, str):
        return response.strip()
    return str(response).strip()

def retrieval_recall_at_k(retrieved_chunks: List[str], gold_snippets: List[Dict], k: int = 10) -> float:
    """
    Calculate retrieval recall@k - fraction of gold snippets found in retrieved chunks.
    
    This measures retrieval quality: how many of the gold answer snippets appear
    in the top-k retrieved chunks from the vector store.
    
    Args:
        retrieved_chunks: List of retrieved chunk text strings.
        gold_snippets: List of gold snippet dictionaries with 'file_path' and 'span' keys.
        k: Number of top chunks to consider (default: 10).
        
    Returns:
        Fraction of gold snippets found in retrieved chunks (0.0 to 1.0).
    """
    if not gold_snippets:
        return 0.0
    
    # Get gold text content from corpus files
    gold_texts = []
    corpus_base = BASE_DIR / "data" / "corpus"
    for snippet in gold_snippets:
        filepath = corpus_base / snippet["file_path"]
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    span = snippet["span"]
                    gold_text = text[span[0]:span[1]]
                    gold_texts.append(gold_text.lower())
            except Exception:
                continue
    
    if not gold_texts:
        return 0.0
    
    if not retrieved_chunks:
        return 0.0
    
    # Filter out empty chunks
    retrieved_chunks = [chunk for chunk in retrieved_chunks[:k] if chunk and chunk.strip()]
    if not retrieved_chunks:
        return 0.0
    
    found_count = 0
    for gold_text in gold_texts:
        if not gold_text or not gold_text.strip():
            continue
            
        # Check if the FULL gold text appears in at least one retrieved chunk
        # This is stricter than checking just the first 100 chars across all chunks
        found = any(gold_text in chunk.lower() for chunk in retrieved_chunks)
        
        # If full text not found and gold text is long (>100 chars), check if substantial portion (90%) appears
        if not found and len(gold_text) > 100:
            min_length = int(len(gold_text) * 0.9)
            if min_length > 0:
                for chunk in retrieved_chunks:
                    chunk_lower = chunk.lower()
                    # Check if any substring of at least min_length from gold_text appears in chunk
                    for i in range(len(gold_text) - min_length + 1):
                        substring = gold_text[i:i + min_length]
                        if substring in chunk_lower:
                            found = True
                            break
                    if found:
                        break
        
        if found:
            found_count += 1
    
    return found_count / len(gold_texts) if gold_texts else 0.0

def simple_date_extractor(text: str) -> str:
    """
    Extract date expressions from text using regex patterns.
    
    Searches for common date formats including:
    - MM/DD/YYYY or MM-DD-YYYY
    - YYYY-MM-DD
    - Month name format (e.g., "January 1, 2024")
    
    Args:
        text: Text string to search for dates.
        
    Returns:
        Comma-separated string of all unique dates found, or "No dates found" if none.
    """
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text, flags=re.IGNORECASE))
    # dedupe
    seen = set()
    out = []
    for f in found:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return ", ".join(out) if out else "No dates found"

# -------------------------------
# Tool Usage Tracking
# -------------------------------
# Per-query tool usage tracking
_current_query_tools = set()
_current_query_retrieved_chunks = []  # Track retrieved chunks for recall@k calculation

def mk_rag_tool(index, llm):
    """
    Create a RAG (Retrieval-Augmented Generation) tool for querying contracts.
    
    Wraps a LlamaIndex query engine as a FunctionTool that can be used by an agent.
    The tool retrieves relevant contract chunks and generates answers using the LLM.
    
    Args:
        index: LlamaIndex VectorStoreIndex containing embedded contract documents.
        llm: Language model instance for answer generation.
        
    Returns:
        A LlamaIndex FunctionTool that takes a query string and returns an answer.
    """
    # Retrieve more chunks to allow meaningful recall@k evaluation
    # Default to 100, but can be overridden via RETRIEVAL_TOP_K env var
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "100"))
    query_engine = index.as_query_engine(similarity_top_k=retrieval_top_k, llm=llm)

    def rag_qa(query: str) -> str:
        """
        Retrieves and answers questions using contract documents via RAG (Retrieval-Augmented Generation).
        
        Use this tool when:
        - You need to find information from contracts
        - The query asks about clauses, terms, provisions, or general contract information
        - You need to answer questions about contract content, legal terms, or obligations
        - The question is NOT specifically about dates (use extract_dates for date questions)
        
        This tool searches through the contract corpus, retrieves relevant chunks,
        and generates an answer using the language model.
        
        Args:
            query: Your question about the contract (e.g., "What is the governing law?",
                   "Is there a non-compete clause?", "What are the termination conditions?")
        
        Returns:
            An answer string based on relevant contract information retrieved via RAG.
        """
        global _current_query_retrieved_chunks
        _current_query_tools.add("rag_qa")
        try:
            response = query_engine.query(query)
            # Extract retrieved chunks from source nodes for recall@k calculation
            if hasattr(response, 'source_nodes'):
                chunks = [node.text for node in response.source_nodes]
                _current_query_retrieved_chunks.extend(chunks)
            if hasattr(response, 'response'):
                return str(response.response)
            return str(response)
        except Exception as e:
            return f"Error in rag_qa: {e}"
    
    return FunctionTool.from_defaults(
        fn=rag_qa,
        name="rag_qa",
        description=rag_qa.__doc__
    )

def mk_extract_dates_tool(index):
    """
    Create an extract_dates tool that can work with queries or text.
    
    If input looks like a question, retrieves relevant contract text first,
    then extracts dates. Otherwise, extracts dates directly from the input text.
    
    Args:
        index: LlamaIndex VectorStoreIndex for retrieving contract text when needed.
        
    Returns:
        A LlamaIndex FunctionTool that extracts dates from text or query results.
    """
    def extract_dates(query_or_text: str) -> str:
        """
        Extracts date expressions from contract text or queries.
        
        Use this tool when the query asks about:
        - Expiration dates (e.g., "What is the expiration date?")
        - Renewal terms (e.g., "What is the renewal term?")
        - Deadlines (e.g., "What is the deadline?")
        - Time periods (e.g., "What is the notice period?")
        - Contract terms related to dates or time
        
        This tool will:
        1. If input looks like a question, retrieve relevant contract text first
        2. Extract all date expressions using regex patterns
        3. Return a comma-separated list of dates found
        
        Args:
            query_or_text: Either a question about dates (e.g., "What is the expiration date?")
                          or contract text containing dates to extract from.
        
        Returns:
            A comma-separated string of all unique dates found in the format:
            "MM/DD/YYYY, YYYY-MM-DD, January 1, 2024, ..."
            Returns "No dates found" if no dates are detected.
        """
        global _current_query_retrieved_chunks
        _current_query_tools.add("extract_dates")
        # If it looks like a question, retrieve text first
        if any(w in query_or_text.lower() for w in ["what", "when", "which", "expiration", "renewal", "deadline", "term"]):
            # Retrieve more chunks to allow meaningful recall@k evaluation
            # Default to 100, but can be overridden via RETRIEVAL_TOP_K env var
            retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "100"))
            query_engine = index.as_query_engine(similarity_top_k=retrieval_top_k)
            response = query_engine.query(query_or_text)
            # Extract retrieved chunks from source nodes for recall@k calculation
            if hasattr(response, 'source_nodes'):
                chunks = [node.text for node in response.source_nodes]
                _current_query_retrieved_chunks.extend(chunks)
            if hasattr(response, 'response'):
                text = str(response.response)
            else:
                text = str(response)
            # Also get source nodes for more context
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    text += "\n\n" + node.text
            return simple_date_extractor(text)
        else:
            return simple_date_extractor(query_or_text)
    
    return FunctionTool.from_defaults(
        fn=extract_dates,
        name="extract_dates",
        description=extract_dates.__doc__
    )

# -------------------------------
# Agent Loader
# -------------------------------
def load_agent():
    idx_dir = BASE_DIR / "data" / "vectorstores" / "llamaindex_rag"
    if not idx_dir.exists():
        raise FileNotFoundError(f"Missing LlamaIndex index at {idx_dir}")

    storage_context = StorageContext.from_defaults(persist_dir=str(idx_dir))
    index = load_index_from_storage(storage_context)

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        mk_extract_dates_tool(index),
        mk_rag_tool(index, llm),
    ]

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=False,
        system_prompt=(
            "You are a legal contract analysis assistant. "
            "For date-based questions use extract_dates(), for everything else use rag_qa()."
        )
    )
    return agent

# -------------------------------
# Async Eval Helper
# -------------------------------
async def ask_agent(agent, query: str, timeout=120):
    try:
        return await asyncio.wait_for(agent.run(query), timeout)
    except asyncio.TimeoutError:
        return None

# -------------------------------
# Evaluation
# -------------------------------
async def evaluate_agent(agent, test_cases: List[Dict], recall_k: int = 10):
    global _current_query_tools, _current_query_retrieved_chunks
    results = {
        "f1_scores": [],
        "latencies": [],
        "tool_efficiency": [],
        "tool_usage_breakdown": {"extract_dates": 0, "rag_qa": 0},
        "recall_at_k": [],
        "detailed_results": []
    }

    for i, tc in enumerate(test_cases):
        q = tc["query"]
        golds = tc.get("gold_answers", [])
        exp_tools = tc.get("expected_tools", [])
        snippets = tc.get("snippets", [])
        
        print(f"[{i+1}/{len(test_cases)}] Query: {q[:80]}...")

        # Reset tool tracking and retrieved chunks for this query
        _current_query_tools = set()
        _current_query_retrieved_chunks = []

        start = time.time()
        try:
            resp_obj = await ask_agent(agent, q)
            latency = time.time() - start

            answer = extract_final_answer(resp_obj)
            tools_used = list(_current_query_tools)
            retrieved_chunks = list(_current_query_retrieved_chunks)

            best_f1 = max((f1_score(answer, g) for g in golds), default=0.0)
            
            # Calculate retrieval recall@k using retrieved chunks and gold snippets
            snippet_dicts = [
                {
                    "file_path": s.file_path,
                    "span": list(s.span) if isinstance(s.span, tuple) else s.span
                }
                for s in snippets
            ]
            rec = retrieval_recall_at_k(retrieved_chunks, snippet_dicts, k=recall_k)

            exp_set, used_set = set(exp_tools), set(tools_used)
            tool_eff = len(exp_set & used_set) / len(exp_set) if exp_set else 0.0

            # Update tool usage breakdown
            for t in tools_used:
                if t in results["tool_usage_breakdown"]:
                    results["tool_usage_breakdown"][t] += 1

            # record
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_eff)
            results["recall_at_k"].append(rec)
            results["detailed_results"].append({
                "query": q,
                "answer": answer,
                "gold_answers": golds,
                "tools_used": tools_used,
                "expected_tools": exp_tools,
                "f1": best_f1,
                "latency": latency,
                "tool_efficiency": tool_eff,
                "recall_at_k": rec,
                "retrieved_chunks_count": len(retrieved_chunks),
            })
        except Exception as e:
            latency = time.time() - start
            print(f"  ERROR: {e}")
            
            best_f1 = 0.0
            tool_eff = 0.0
            rec = 0.0
            tools_used = []
            retrieved_chunks = []
            
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_eff)
            results["recall_at_k"].append(rec)
            results["detailed_results"].append({
                "query": q,
                "answer": "ERROR",
                "gold_answers": golds,
                "tools_used": tools_used,
                "expected_tools": exp_tools,
                "f1": best_f1,
                "latency": latency,
                "tool_efficiency": tool_eff,
                "recall_at_k": rec,
                "retrieved_chunks_count": len(retrieved_chunks),
                "error": str(e)
            })

    return results

# -------------------------------
# MAIN
# -------------------------------
def main():
    if WANDB_AVAILABLE:
        wandb.init(
            project="legal-rag-evaluation",
            name="llamaindex-tool-use-cuad",
            config={
                "framework": "LlamaIndex",
                "mode": "tool-use",
                "dataset": "CUAD",
                "model": "gpt-4o-mini",
                "num_tools": 2,
                "tools": ["extract_dates", "rag_qa"],
                "timeout_per_query": 120,
                "recall_k": int(os.getenv("RECALL_K", "10")),
                "retrieval_top_k": int(os.getenv("RETRIEVAL_TOP_K", "100")),
            }
        )

    with open(BASE_DIR / "data" / "benchmarks" / "cuad.json", "r") as fh:
        benchmark = Benchmark(**json.load(fh))

    # Create test cases with filtering and expected_tools (same as LangChain)
    date_keywords = ['date', 'expiration', 'expires', 'renewal', 'deadline', 'period']
    rag_keywords = ['what', 'how', 'which', 'clause', 'provision', 'term', 'agreement', 'governing', 'law', 'liability', 'warranty', 'termination', 'indemnification', 'right', 'obligation', 'party']
    test_cases = []
    corpus_dir = BASE_DIR / "data" / "corpus"
    
    for test in benchmark.tests:
        ql = test.query.lower()
        # More restrictive date detection - only if explicitly about dates/time
        needs_dates = any(k in ql for k in date_keywords) and ('date' in ql or 'expir' in ql or 'deadline' in ql or 'renewal' in ql)
        # RAG is needed for most queries - broader keyword matching and default
        needs_rag = any(k in ql for k in rag_keywords) or not needs_dates
        
        if needs_dates or needs_rag:
            golds = []
            for s in test.snippets:
                p = corpus_dir / s.file_path
                if p.exists():
                    txt = p.read_text(encoding="utf-8")
                    golds.append(txt[s.span[0]:s.span[1]])
            
            expected = []
            if needs_dates:
                expected.append("extract_dates")
            if needs_rag:
                expected.append("rag_qa")
            
            test_cases.append({
                "query": test.query,
                "gold_answers": golds,
                "expected_tools": expected,
                "snippets": test.snippets
            })
    
    print(f"Total tool-use candidates: {len(test_cases)}")

    agent = load_agent()
    recall_k = int(os.getenv("RECALL_K", "10"))
    
    # run evaluation (async)
    results = asyncio.run(evaluate_agent(agent, test_cases[:3], recall_k=recall_k))

    # compute summary
    avg_f1 = sum(results["f1_scores"])/len(results["f1_scores"])
    avg_lat = sum(results["latencies"])/len(results["latencies"])
    avg_te = sum(results["tool_efficiency"])/len(results["tool_efficiency"])
    avg_rec = sum(results["recall_at_k"])/len(results["recall_at_k"])

    # Save to disk (same format as LangChain)
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    json_out = out_dir / "llamaindex_tool_use_eval_detailed.json"
    csv_out = out_dir / "llamaindex_tool_use_eval_summary.csv"

    # Get wandb URL before finishing the run
    wandb_url = None
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb_url = wandb.run.url

    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump({
            "summary": {
                "num_cases": len(results["f1_scores"]),
                "f1_score": avg_f1,
                "avg_latency": avg_lat,
                "tool_efficiency": avg_te,
                "recall_at_k": avg_rec,
                "tool_usage_breakdown": results["tool_usage_breakdown"],
                "wandb_run_url": wandb_url,
            },
            "detailed_results": results["detailed_results"]
        }, jf, ensure_ascii=False, indent=2)

    with open(csv_out, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["case", "query", "answer_snippet", "f1", "tools_used", "expected_tools", "tool_efficiency", "recall_at_k", "latency"])
        for idx, dr in enumerate(results["detailed_results"]):
            w.writerow([
                idx + 1,
                (dr.get("query") or "")[:400],
                (dr.get("answer") or "")[:500],
                dr.get("f1", 0.0),
                ";".join(dr.get("tools_used", [])),
                ";".join(dr.get("expected_tools", [])),
                dr.get("tool_efficiency", 0.0),
                dr.get("recall_at_k", 0.0),
                dr.get("latency", 0.0),
            ])

    # W&B logging
    if WANDB_AVAILABLE:
        wandb.log({
            "f1_score": avg_f1,
            "avg_latency": avg_lat,
            "tool_efficiency": avg_te,
            "recall_at_k": avg_rec,
            "num_test_cases": len(results["f1_scores"]),
            "tool_usage_extract_dates": results["tool_usage_breakdown"]["extract_dates"],
            "tool_usage_rag_qa": results["tool_usage_breakdown"]["rag_qa"],
            "f1_distribution": wandb.Histogram(results["f1_scores"]),
            "latency_distribution": wandb.Histogram(results["latencies"]),
            "tool_efficiency_distribution": wandb.Histogram(results["tool_efficiency"]),
        })
        print(f"\nResults logged to wandb: {wandb.run.url}")
        wandb.finish()

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"F1 Score: {avg_f1:.2%}")
    print(f"Avg Latency: {avg_lat:.2f}s")
    print(f"Tool Efficiency: {avg_te:.2%}")
    print(f"Recall@k: {avg_rec:.2%}")
    print(f"Tool Usage: extract_dates={results['tool_usage_breakdown']['extract_dates']}, rag_qa={results['tool_usage_breakdown']['rag_qa']}")
    print(f"\nResults saved to:")
    print(f"  JSON: {json_out}")
    print(f"  CSV: {csv_out}")
    print("="*50)

if __name__ == "__main__":
    main()
