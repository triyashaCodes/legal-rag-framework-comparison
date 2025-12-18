#!/usr/bin/env python3
"""
Evaluate Response Quality Metrics for RAG Systems

Evaluates how close generated answers are to gold legal spans:
- F1 Score: Token-level overlap between generated answer and gold spans
- BLEU: N-gram precision-based metric for lexical similarity
- ROUGE-1: Unigram recall, measures content coverage
- ROUGE-2: Bigram overlap, captures phrase-level alignment
- ROUGE-L: Longest Common Subsequence, measures sentence-level structure similarity
- BARTScore: Semantic similarity metric using pretrained BART model
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Optional: nltk for BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not available. BLEU score will be disabled.")

# Optional: rouge-score for ROUGE metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be disabled.")

# Optional: bart-score for BARTScore
try:
    from bart_score import BARTScorer
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    print("Warning: bart-score not available. BARTScore will be disabled.")

# -------------------------------
# ENV / PATH
# -------------------------------
if "google.colab" in sys.modules:
    BASE_DIR = Path("/content/CUAD-RAG")
else:
    BASE_DIR = Path(__file__).parent

# -------------------------------
# METRICS
# -------------------------------
def f1_score(predicted: str, gold: str) -> float:
    """Token-level F1 score for factual accuracy."""
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

def bleu_score(predicted: str, gold: str) -> float:
    """
    Calculate BLEU score - N-gram precision-based metric for lexical similarity.
    Captures lexical similarity, useful for short, structured legal answers.
    
    Args:
        predicted: The predicted answer string.
        gold: The gold/ground truth answer string.
        
    Returns:
        BLEU score between 0.0 and 1.0 (higher is better).
    """
    if not BLEU_AVAILABLE:
        return 0.0
    
    try:
        pred_tokens = predicted.lower().split()
        gold_tokens = gold.lower().split()
        
        if not gold_tokens:
            return 1.0 if not pred_tokens else 0.0
        if not pred_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothing)
        return float(score)
    except Exception:
        return 0.0

def rouge_scores(predicted: str, gold: str) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    - ROUGE-1: Unigram recall, measures content coverage
    - ROUGE-2: Bigram overlap, captures phrase-level alignment
    - ROUGE-L: Longest Common Subsequence, measures sentence-level structure similarity
    
    Args:
        predicted: The predicted answer string.
        gold: The gold/ground truth answer string.
        
    Returns:
        Dictionary with 'rouge1', 'rouge2', 'rougel' scores (f-measure).
    """
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougel": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gold, predicted)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougel": scores['rougeL'].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougel": 0.0}

# Global BART scorer instance (lazy initialization)
_bart_scorer = None

def bart_score(predicted: str, gold: str) -> float:
    """
    Calculate BARTScore - semantic similarity metric using pretrained BART model.
    Captures meaning alignment, not just token overlap.
    Important because legal answers may be paraphrased.
    
    Args:
        predicted: The predicted answer string.
        gold: The gold/ground truth answer string.
        
    Returns:
        BARTScore normalized to 0-1 range (higher is better).
    """
    global _bart_scorer
    
    if not BART_AVAILABLE:
        return 0.0
    
    try:
        if _bart_scorer is None:
            _bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
        
        # BARTScore returns negative values (higher is better)
        # We'll normalize to 0-1 range for consistency
        score = _bart_scorer.score([predicted], [gold], batch_size=1)[0]
        # Normalize: assume scores range from -5 to 0, map to 0-1
        normalized = max(0.0, min(1.0, (score + 5) / 5)) if score < 0 else 1.0
        return normalized
    except Exception:
        return 0.0

# -------------------------------
# MAIN EVALUATION FUNCTION
# -------------------------------
def evaluate_response_quality(
    results_file: str,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Evaluate response quality from evaluation results.
    
    Calculates F1, BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and BARTScore metrics
    comparing generated answers to gold legal spans.
    
    Args:
        results_file: Path to JSON file with evaluation results
        output_file: Optional path to save detailed results
    
    Returns:
        Dictionary with quality metrics
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract test cases
    test_cases = results.get("detailed_results", [])
    if not test_cases:
        print("Error: No detailed results found in file")
        return {}
    
    quality_metrics = {
        "f1_scores": [],
        "bleu_scores": [],
        "rouge1_scores": [],
        "rouge2_scores": [],
        "rougel_scores": [],
        "bart_scores": [],
        "detailed_quality": []
    }
    
    for i, tc in enumerate(test_cases):
        answer = tc.get("answer", "")
        gold_answers = tc.get("gold_answers", [])
        
        if not answer or not gold_answers:
            continue
        
        # F1 score (factual accuracy)
        best_f1 = max((f1_score(answer, g) for g in gold_answers), default=0.0)
        
        # BLEU score (lexical similarity)
        best_bleu = max((bleu_score(answer, g) for g in gold_answers), default=0.0)
        
        # ROUGE scores (content coverage and alignment)
        best_rouge1 = 0.0
        best_rouge2 = 0.0
        best_rougel = 0.0
        for g in gold_answers:
            rouge_s = rouge_scores(answer, g)
            best_rouge1 = max(best_rouge1, rouge_s["rouge1"])
            best_rouge2 = max(best_rouge2, rouge_s["rouge2"])
            best_rougel = max(best_rougel, rouge_s["rougel"])
        
        # BARTScore (semantic similarity)
        best_bart = max((bart_score(answer, g) for g in gold_answers), default=0.0)
        
        # Store metrics
        quality_metrics["f1_scores"].append(best_f1)
        quality_metrics["bleu_scores"].append(best_bleu)
        quality_metrics["rouge1_scores"].append(best_rouge1)
        quality_metrics["rouge2_scores"].append(best_rouge2)
        quality_metrics["rougel_scores"].append(best_rougel)
        quality_metrics["bart_scores"].append(best_bart)
        
        quality_metrics["detailed_quality"].append({
            "query": tc.get("query", ""),
            "answer": answer,
            "f1": best_f1,
            "bleu": best_bleu,
            "rouge1": best_rouge1,
            "rouge2": best_rouge2,
            "rougel": best_rougel,
            "bart": best_bart,
        })
    
    # Compute averages
    summary = {
        "avg_f1": sum(quality_metrics["f1_scores"]) / len(quality_metrics["f1_scores"]) if quality_metrics["f1_scores"] else 0.0,
        "avg_bleu": sum(quality_metrics["bleu_scores"]) / len(quality_metrics["bleu_scores"]) if quality_metrics["bleu_scores"] else 0.0,
        "avg_rouge1": sum(quality_metrics["rouge1_scores"]) / len(quality_metrics["rouge1_scores"]) if quality_metrics["rouge1_scores"] else 0.0,
        "avg_rouge2": sum(quality_metrics["rouge2_scores"]) / len(quality_metrics["rouge2_scores"]) if quality_metrics["rouge2_scores"] else 0.0,
        "avg_rougel": sum(quality_metrics["rougel_scores"]) / len(quality_metrics["rougel_scores"]) if quality_metrics["rougel_scores"] else 0.0,
        "avg_bart": sum(quality_metrics["bart_scores"]) / len(quality_metrics["bart_scores"]) if quality_metrics["bart_scores"] else 0.0,
        "num_evaluated": len(quality_metrics["f1_scores"]),
    }
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_quality": quality_metrics["detailed_quality"],
                "all_metrics": quality_metrics
            }, f, indent=2)
        print(f"Quality evaluation results saved to {output_file}")
    
    return {
        "summary": summary,
        "detailed": quality_metrics
    }

# -------------------------------
# MAIN
# -------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate response quality metrics")
    parser.add_argument("results_file", help="Path to evaluation results JSON file")
    parser.add_argument("--output", "-o", help="Output file for quality metrics", default=None)
    
    args = parser.parse_args()
    
    results = evaluate_response_quality(
        args.results_file,
        args.output
    )
    
    if not results:
        return
    
    summary = results["summary"]
    
    # Print summary
    print("\n" + "="*50)
    print("RESPONSE QUALITY EVALUATION SUMMARY")
    print("="*50)
    print(f"Factual Accuracy (F1): {summary['avg_f1']:.2%}")
    print(f"BLEU Score: {summary['avg_bleu']:.4f}")
    print(f"ROUGE-1: {summary['avg_rouge1']:.4f}")
    print(f"ROUGE-2: {summary['avg_rouge2']:.4f}")
    print(f"ROUGE-L: {summary['avg_rougel']:.4f}")
    print(f"BARTScore: {summary['avg_bart']:.4f}")
    print(f"Number of Responses Evaluated: {summary['num_evaluated']}")
    print("="*50)

if __name__ == "__main__":
    main()

