#!/usr/bin/env python3
"""
Evaluate LangChain Tool-Use (Agent) on CUAD benchmark.

Modern evaluator for LangChain create_agent tool-calling agents.
Tests agent's ability to use tools (extract_dates, rag_qa) for legal document analysis.
Saves detailed JSON + CSV results. Compatible with agent.invoke({"input": query}) style.
"""

import os
import json
import time
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
from threading import Thread
import queue
import re

# -------------------------------
# ENV / PATH
# -------------------------------
if 'google.colab' in sys.modules:
    current_dir = Path.cwd()
    BASE_DIR = current_dir if (current_dir / 'generate_cuad.py').exists() else Path('/content')
else:
    BASE_DIR = Path(__file__).parent

# -------------------------------
# IMPORTS (keep these as in your environment)
# -------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_classic.chains import RetrievalQA
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

def extract_final_answer(run_result: Dict[str, Any]) -> str:
    """
    Extract the final answer from agent's run result.
    
    Scans run_result["messages"] from the end and returns the first non-empty
    AI/assistant message content. Falls back to run_result.get("output") if present.
    
    Args:
        run_result: Dictionary containing agent execution result with "messages" or "output" key.
        
    Returns:
        The final answer string, or empty string if no answer found.
    """
    msgs = run_result.get("messages", []) if isinstance(run_result, dict) else []
    for m in reversed(msgs):
        # m might be dict-like or object-like
        content = None
        if isinstance(m, dict):
            content = m.get("content") or m.get("text")
            role = m.get("role") or m.get("type") or ""
        else:
            content = getattr(m, "content", None)
            role = getattr(m, "type", "") or getattr(m, "role", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    # fallback
    if isinstance(run_result, dict):
        out = run_result.get("output")
        if isinstance(out, str) and out.strip():
            return out.strip()
    return ""

def extract_tool_calls(run_result: Dict[str, Any]) -> List[str]:
    """
    Extract tool names that were called during agent execution.
    
    Parses run_result["messages"] for tool_calls stored in additional_kwargs["tool_calls"].
    Falls back to searching message content for tool keywords if structured data not available.
    
    Args:
        run_result: Dictionary containing agent execution result with "messages" key.
        
    Returns:
        List of unique tool names that were called, in order of appearance.
    """
    msgs = run_result.get("messages", []) if isinstance(run_result, dict) else []
    tool_names: List[str] = []
    for m in msgs:
        additional_kwargs = {}
        content = ""
        if isinstance(m, dict):
            additional_kwargs = m.get("additional_kwargs", {}) or {}
            content = m.get("content", "") or ""
        else:
            additional_kwargs = getattr(m, "additional_kwargs", {}) or {}
            content = getattr(m, "content", "") or ""
        tc_list = additional_kwargs.get("tool_calls", []) or additional_kwargs.get("tool_call", []) or []
        for tc in tc_list:
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("tool") or tc.get("tool_name")
                if name:
                    tool_names.append(name)
            else:
                name = getattr(tc, "name", None) or getattr(tc, "tool", None)
                if name:
                    tool_names.append(name)
        # fallback content scanning (best-effort)
        if isinstance(content, str):
            if "extract_dates" in content.lower():
                tool_names.append("extract_dates")
            if "rag_qa" in content.lower() or "rag" in content.lower():
                tool_names.append("rag_qa")
    # dedupe preserving order
    seen = set()
    dedup = []
    for t in tool_names:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup

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
# Tool creators (wrapers)
# -------------------------------
def create_rag_tool(vectorstore, llm):
    """
    Create a RAG (Retrieval-Augmented Generation) tool for querying contracts.
    
    Wraps a RetrievalQA chain as a LangChain tool that can be used by an agent.
    The tool retrieves relevant contract chunks and generates answers using the LLM.
    
    Args:
        vectorstore: FAISS vector store containing embedded contract documents.
        llm: Language model instance for answer generation.
        
    Returns:
        A LangChain tool function that takes a query string and returns an answer.
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )

    @tool
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
        try:
            res = qa_chain.invoke({"query": query})
            if isinstance(res, dict):
                return res.get("result", str(res))
            return str(res)
        except Exception as e:
            return f"Error in rag_qa: {e}"
    return rag_qa

def create_extract_dates_tool(vectorstore):
    """
    Create an extract_dates tool that can work with queries or text.
    
    If input looks like a question, retrieves relevant contract text first,
    then extracts dates. Otherwise, extracts dates directly from the input text.
    
    Args:
        vectorstore: FAISS vector store for retrieving contract text when needed.
        
    Returns:
        A LangChain tool function that extracts dates from text or query results.
    """
    @tool
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
        # If it looks like a question, retrieve text first
        if any(w in query_or_text.lower() for w in ["what", "when", "which", "expiration", "renewal", "deadline", "term"]):
            docs = vectorstore.similarity_search(query_or_text, k=10)
            text = "\n\n".join([d.page_content for d in docs])
            return simple_date_extractor(text)
        else:
            return simple_date_extractor(query_or_text)
    return extract_dates

# -------------------------------
# Agent loader
# -------------------------------
def load_tool_use_agent():
    """
    Load and initialize the tool-use agent with RAG and date extraction tools.
    
    Loads the FAISS vector store, creates RAG and extract_dates tools,
    and initializes a LangChain agent with these tools.
    
    Returns:
        Initialized LangChain agent with tools attached.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore_path = BASE_DIR / "data" / "vectorstores" / "langchain_faiss"
    vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_tool = create_rag_tool(vectorstore, llm)
    extract_dates_tool = create_extract_dates_tool(vectorstore)
    tools = [extract_dates_tool, rag_tool]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a legal analysis assistant specialized in contract analysis.\n\n"
            "Tool Usage Guidelines:\n"
            "1. For questions about DATES (expiration, renewal, deadlines, terms, periods):\n"
            "   - Use extract_dates tool directly with the query\n"
            "   - The tool will automatically retrieve relevant contract text and extract dates\n"
            "   - Example queries: 'What is the expiration date?', 'What is the renewal term?', 'When does this expire?'\n\n"
            "2. For other contract questions (clauses, provisions, terms, general information):\n"
            "   - Use rag_qa tool with the query\n"
            "   - Example queries: 'What is the governing law?', 'Is there a non-compete clause?', 'What are the termination conditions?'\n\n"
            "3. Always provide a final, concise answer after using tools. Do not call tools repeatedly.\n"
            "4. If a tool returns an error or no results, explain that in your final answer."
        )
    )
    agent._vectorstore = vectorstore
    return agent

def safe_invoke(agent, query: str, timeout: int = 120):
    """
    Safely invoke agent with timeout protection.
    
    Runs agent.invoke() in a separate thread with a timeout to prevent
    infinite loops or hanging. Raises TimeoutError if execution exceeds timeout.
    
    Args:
        agent: LangChain agent instance to invoke.
        query: Query string to pass to the agent.
        timeout: Maximum execution time in seconds (default: 120).
        
    Returns:
        Agent execution result dictionary.
        
    Raises:
        TimeoutError: If agent execution exceeds timeout.
        RuntimeError: If agent returns no result.
        Exception: Any exception raised during agent execution.
    """
    q_res = queue.Queue()
    q_err = queue.Queue()
    def runner():
        try:
            out = agent.invoke({"input": query})
            q_res.put(out)
        except Exception as e:
            q_err.put(e)
    th = Thread(target=runner, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError("Agent timed out")
    if not q_err.empty():
        raise q_err.get()
    if q_res.empty():
        raise RuntimeError("Agent returned nothing")
    return q_res.get()

def create_tool_use_test_cases(benchmark: Benchmark) -> List[Dict]:
    """
    Create test cases from benchmark that benefit from tool use.
    
    Filters benchmark tests to identify cases that would benefit from
    extract_dates or rag_qa tools based on query keywords.
    
    Args:
        benchmark: Benchmark object containing test cases with queries and snippets.
        
    Returns:
        List of dictionaries, each containing:
        - query: The test query
        - gold_answers: List of gold answer texts
        - expected_tools: List of tool names that should be used
        - snippets: Original snippet objects
    """
    date_keywords = ['date', 'expiration', 'expires', 'renewal', 'deadline', 'period']
    rag_keywords = ['what', 'how', 'which', 'clause', 'provision', 'term', 'agreement', 'governing', 'law', 'liability', 'warranty', 'termination', 'indemnification', 'right', 'obligation', 'party']
    cases = []
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
                    with open(p, "r", encoding="utf-8") as fh:
                        t = fh.read()
                        golds.append(t[s.span[0]:s.span[1]])
            expected = []
            if needs_dates:
                expected.append("extract_dates")
            if needs_rag:
                expected.append("rag_qa")
            cases.append({"query": test.query, "gold_answers": golds, "expected_tools": expected, "snippets": test.snippets})
    return cases

def evaluate_tool_use(agent, test_cases: List[Dict], max_cases: int = None, recall_k: int = 3, timeout: int = 120) -> Dict[str, Any]:
    """
    Evaluate tool-use agent on test cases.
    
    Runs the agent on each test case, extracts answers and tool usage,
    calculates metrics (F1, tool efficiency, recall@k), and saves results to JSON and CSV.
    
    Args:
        agent: LangChain agent with tools to evaluate.
        test_cases: List of test case dictionaries with query, gold_answers, expected_tools.
        max_cases: Maximum number of test cases to evaluate (None for all).
        recall_k: K value for recall@k calculation (default: 3).
        timeout: Timeout per query in seconds (default: 120).
        
    Returns:
        Dictionary containing:
        - f1_scores: List of F1 scores per test case
        - latencies: List of latencies per test case
        - tool_efficiency: List of tool efficiency scores
        - tool_usage_breakdown: Dict counting tool usage
        - detailed_results: List of detailed per-case results
        - recall_at_k: List of recall@k scores
    """
    if max_cases:
        test_cases = test_cases[:max_cases]

    results = {
        "f1_scores": [],
        "latencies": [],
        "tool_efficiency": [],
        "tool_usage_breakdown": {"extract_dates": 0, "rag_qa": 0},
        "detailed_results": [],
        "recall_at_k": []
    }

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        expected_tools = tc.get("expected_tools", [])
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_cases)} cases...")

        start = time.time()
        try:
            # LLM call
            run_out = safe_invoke(agent, query, timeout=timeout)
            latency = time.time() - start
            answer = extract_final_answer(run_out)
            tools_used = extract_tool_calls(run_out)

            gold_texts = tc.get("gold_answers", [])
            
            # Calculate real metrics
            best_f1 = 0.0
            for g in gold_texts:
                best_f1 = max(best_f1, f1_score(answer, g))
            
            expected_set = set(expected_tools)
            used_set = set(tools_used)
            tool_eff = len(expected_set & used_set) / len(expected_set) if expected_set else 0.0

            recall_hits = sum(1 for g in gold_texts if g.lower() in answer.lower())
            recall = min(1.0, recall_hits / (recall_k if recall_k > 0 else 1))

            # update breakdown
            for t in tools_used:
                if t in results["tool_usage_breakdown"]:
                    results["tool_usage_breakdown"][t] += 1

            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_eff)
            results["recall_at_k"].append(recall)
            results["detailed_results"].append({
                "query": query,
                "answer": answer,
                "gold_answers": gold_texts,
                "f1": best_f1,
                "latency": latency,
                "tools_used": tools_used,
                "expected_tools": expected_tools,
                "tool_efficiency": tool_eff,
                "recall_at_k": recall,
            })

        except TimeoutError as e:
            latency = time.time() - start
            print("  TIMEOUT")
            
            best_f1 = 0.0
            tool_eff = 0.0
            recall = 0.0
            tools_used = []
            
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_eff)
            results["recall_at_k"].append(recall)
            results["detailed_results"].append({
                "query": query,
                "answer": "TIMEOUT",
                "f1": best_f1,
                "latency": latency,
                "tools_used": tools_used,
                "expected_tools": expected_tools,
                "tool_efficiency": tool_eff,
                "recall_at_k": recall,
                "error": str(e)
            })

        except Exception as e:
            latency = time.time() - start
            print("  ERROR:", e)
            
            best_f1 = 0.0
            tool_eff = 0.0
            recall = 0.0
            tools_used = []
            
            results["f1_scores"].append(best_f1)
            results["latencies"].append(latency)
            results["tool_efficiency"].append(tool_eff)
            results["recall_at_k"].append(recall)
            results["detailed_results"].append({
                "query": query,
                "answer": "ERROR",
                "f1": best_f1,
                "latency": latency,
                "tools_used": tools_used,
                "expected_tools": expected_tools,
                "tool_efficiency": tool_eff,
                "recall_at_k": recall,
                "error": str(e)
            })

    # Calculate summary metrics
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"]) if results["f1_scores"] else 0.0
    avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0.0
    avg_tool_eff = sum(results["tool_efficiency"]) / len(results["tool_efficiency"]) if results["tool_efficiency"] else 0.0
    avg_recall = sum(results["recall_at_k"]) / len(results["recall_at_k"]) if results["recall_at_k"] else 0.0
    
    # Store summary metrics in results for logging in main()
    results["summary_metrics"] = {
        "avg_f1": avg_f1,
        "avg_latency": avg_latency,
        "avg_tool_eff": avg_tool_eff,
        "avg_recall": avg_recall,
    }
    
    # Save to disk
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    json_out = out_dir / "langchain_tool_use_eval_detailed.json"
    csv_out = out_dir / "langchain_tool_use_eval_summary.csv"

    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump({
            "summary": {
                "num_cases": len(results["f1_scores"]),
                "f1_score": avg_f1,
                "avg_latency": avg_latency,
                "tool_efficiency": avg_tool_eff,
                "recall_at_k": avg_recall,
                "tool_usage_breakdown": results["tool_usage_breakdown"],
                "wandb_run_url": wandb.run.url if WANDB_AVAILABLE else None,
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

    return results

# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main evaluation function.
    
    Loads CUAD benchmark, creates tool-use test cases, initializes agent,
    runs evaluation, logs to wandb, and prints summary statistics.
    """
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="legal-rag-evaluation",
            name="langchain-tool-use-cuad",
            config={
                "framework": "LangChain",
                "mode": "tool-use",
                "dataset": "CUAD",
                "model": "gpt-4o-mini",
                "num_tools": 2,
                "tools": ["extract_dates", "rag_qa"],
                "timeout_per_query": 120,
                "recall_k": int(os.getenv("RECALL_K", "10")),
            }
        )
    
    total_start_time = time.time()
    
    bench_path = BASE_DIR / "data" / "benchmarks" / "cuad.json"
    with open(bench_path, "r", encoding="utf-8") as fh:
        bench_data = json.load(fh)
    benchmark = Benchmark(**bench_data)

    test_cases = create_tool_use_test_cases(benchmark)
    print(f"Total tool-use candidates: {len(test_cases)}")

    agent = load_tool_use_agent()
    # Allow recall_k to be set via environment variable, default to 10
    recall_k = int(os.getenv("RECALL_K", "10"))
    results = evaluate_tool_use(agent, test_cases, max_cases=3, recall_k=recall_k, timeout=120)
    
    total_time = time.time() - total_start_time

    # Summary print
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"]) if results["f1_scores"] else 0.0
    avg_lat = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0.0
    avg_te = sum(results["tool_efficiency"]) / len(results["tool_efficiency"]) if results["tool_efficiency"] else 0.0
    avg_rec = sum(results["recall_at_k"]) / len(results["recall_at_k"]) if results["recall_at_k"] else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"F1 Score: {avg_f1:.2%}")
    print(f"Avg Latency: {avg_lat:.2f}s")
    print(f"Tool Efficiency: {avg_te:.2%}")
    print(f"Recall@k: {avg_rec:.2%}")
    print(f"Total Evaluation Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Tool Usage: extract_dates={results['tool_usage_breakdown']['extract_dates']}, rag_qa={results['tool_usage_breakdown']['rag_qa']}")
    print("="*50)
    
    if WANDB_AVAILABLE:
        # Log all metrics together in one call
        wandb.log({
            "f1_score": results["summary_metrics"]["avg_f1"],
            "avg_latency": results["summary_metrics"]["avg_latency"],
            "tool_efficiency": results["summary_metrics"]["avg_tool_eff"],
            "recall_at_k": results["summary_metrics"]["avg_recall"],
            "num_test_cases": len(results["f1_scores"]),
            "tool_usage_extract_dates": results["tool_usage_breakdown"]["extract_dates"],
            "tool_usage_rag_qa": results["tool_usage_breakdown"]["rag_qa"],
            "total_evaluation_time": total_time,
            "f1_distribution": wandb.Histogram(results["f1_scores"]),
            "latency_distribution": wandb.Histogram(results["latencies"]),
            "tool_efficiency_distribution": wandb.Histogram(results["tool_efficiency"]),
        })
        print(f"\nResults logged to wandb: {wandb.run.url}")
        wandb.finish()

if __name__ == "__main__":
    main()
