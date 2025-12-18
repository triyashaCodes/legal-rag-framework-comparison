# Framework Comparison for Legal RAG Systems: LangChain vs LlamaIndex

This repository contains the code, data, and results for the paper "Framework Comparison for Legal RAG Systems: LangChain vs LlamaIndex" evaluating both frameworks on the CUAD-RAG benchmark.

## Overview

This study provides a systematic comparison of LangChain and LlamaIndex frameworks for legal Retrieval-Augmented Generation (RAG) systems, evaluating both simple RAG pipelines and tool-use agents on the Contract Understanding Atticus Dataset (CUAD).

## Repository Structure

```
legal-rag-framework-comparison/
├── src/                    # Source code
│   ├── evaluate_*.py      # Evaluation scripts
│   │   ├── evaluate_langchain_tool_use.py    # LangChain tool-use evaluation
│   │   ├── evaluate_llamaindex_tool_use.py   # LlamaIndex tool-use evaluation
│   │   └── evaluate_response_quality.py     # Response quality metrics (BLEU/ROUGE/BARTScore)
│   ├── langchain_*.py      # LangChain implementations
│   ├── llamaindex_*.py    # LlamaIndex implementations
│   ├── generate_cuad.py   # CUAD benchmark generation
│   ├── benchmark_types.py # Benchmark data structures
│   └── utils.py           # Utility functions
├── data/
│   ├── benchmarks/        # CUAD-RAG benchmark
│   ├── corpus/            # CUAD dataset (user-provided)
│   └── vectorstores/      # Generated vector stores
├── figures/                # Generated figures (PDF and PNG)
├── results/                # Experimental results (CSV exports)
├── paper/                  
└── requirements.txt        # Python dependencies
```

## Prerequisites

- Python 3.9 or higher
- OpenAI API key with access to GPT-3.5-turbo and GPT-4o-mini
- CUAD dataset (download separately)
- ~50GB disk space for vector stores and data
- WandB account (optional, for experiment tracking)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd legal-rag-framework-comparison
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

**Option 1: Using a `.env` file**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key
WANDB_API_KEY=your-wandb-key  # Optional
RECALL_K=10
RETRIEVAL_TOP_K=100
```

Get your API keys:
- `OPENAI_API_KEY`: Required - Get from https://platform.openai.com/api-keys
- `WANDB_API_KEY`: Optional - Get from https://wandb.ai/authorize

Then load it before running scripts:
```bash
export $(cat .env | xargs)
```

**Option 2: Export directly**

```bash
export OPENAI_API_KEY="your-api-key"
export WANDB_API_KEY="your-wandb-key"  # Optional
```

### 5. Environment Variables for Evaluation

For tool-use agent evaluations, you can configure retrieval and recall parameters:

- **`RECALL_K`**: Number of top-k documents to consider for recall calculation (default: `10`)
- **`RETRIEVAL_TOP_K`**: Number of documents to retrieve from the vector store (default: `100`)

**Examples:**

```bash
# Set recall evaluation parameter
export RECALL_K=10

# Set retrieval parameter (LlamaIndex only)
export RETRIEVAL_TOP_K=100

# Run evaluation
python evaluate_langchain_tool_use.py
# or
python evaluate_llamaindex_tool_use.py
```

## Data Setup

### CUAD Dataset

1. Download CUAD v1 from [The Atticus Project](https://github.com/TheAtticusProject/cuad)
2. Extract the dataset
3. Place contract text files in: `data/corpus/cuad/`

The structure should be:
```
data/
├── corpus/
│   └── cuad/
│       ├── contract1.txt
│       ├── contract2.txt
│       └── ...
└── benchmarks/
    └── cuad.json
```

**Note**: Due to dataset licensing, we provide only the benchmark structure. Users must obtain the CUAD dataset separately.

### Generate Benchmark

```bash
cd src
python generate_cuad.py
```

This will create `data/benchmarks/cuad.json` with query-answer pairs.

## Usage

### Step 1: Build Vector Stores

**LangChain:**
```bash
cd src
python langchain_rag.py
```

**LlamaIndex:**
```bash
python llamaindex_rag.py
```

Vector stores will be saved in `data/vectorstores/`.

### Step 2: Run Evaluations

**Simple RAG Evaluation:**
```bash
python evaluate_langchain.py
python evaluate_llamaindex_rag.py
```

**Tool-Use Agent Evaluation:**

**LangChain (k=10, 20, 100):**
```bash
export RECALL_K=10
python evaluate_langchain_tool_use.py

export RECALL_K=20
python evaluate_langchain_tool_use.py

export RECALL_K=100
python evaluate_langchain_tool_use.py
```

Results are saved to:
- `results/langchain_tool_use_eval_detailed.json` - Detailed results with answers and metrics
- `results/langchain_tool_use_eval_summary.csv` - Summary CSV export

**LlamaIndex (k=10, 20, 100):**
```bash
export RECALL_K=10
export RETRIEVAL_TOP_K=20
python evaluate_llamaindex_tool_use.py

export RECALL_K=20
export RETRIEVAL_TOP_K=50
python evaluate_llamaindex_tool_use.py

export RECALL_K=100
export RETRIEVAL_TOP_K=500
python evaluate_llamaindex_tool_use.py
```

Results are saved to:
- `results/llamaindex_tool_use_eval_detailed.json` - Detailed results with answers and metrics
- `results/llamaindex_tool_use_eval_summary.csv` - Summary CSV export

### Step 3: Evaluate Response Quality

After running tool-use evaluations, you can evaluate response quality metrics (BLEU, ROUGE, BARTScore) on the generated results:

```bash
# Evaluate LangChain results
python evaluate_response_quality.py results/langchain_tool_use_eval_detailed.json --output results/langchain_quality_metrics.json

# Evaluate LlamaIndex results
python evaluate_response_quality.py results/llamaindex_tool_use_eval_detailed.json --output results/llamaindex_quality_metrics.json
```

This script calculates:
- **F1 Score**: Token-level overlap between generated answer and gold spans
- **BLEU**: N-gram precision-based metric for lexical similarity
- **ROUGE-1**: Unigram recall, measures content coverage
- **ROUGE-2**: Bigram overlap, captures phrase-level alignment
- **ROUGE-L**: Longest Common Subsequence, measures sentence-level structure similarity
- **BARTScore**: Semantic similarity metric using pretrained BART model


## Results

Experimental results are saved in the `results/` directory:

**Tool-Use Evaluation Results:**
- `langchain_tool_use_eval_detailed.json` - LangChain detailed results (JSON)
- `langchain_tool_use_eval_summary.csv` - LangChain summary (CSV)
- `llamaindex_tool_use_eval_detailed.json` - LlamaIndex detailed results (JSON)
- `llamaindex_tool_use_eval_summary.csv` - LlamaIndex summary (CSV)

**Response Quality Metrics:**
- `langchain_quality_metrics.json` - LangChain BLEU/ROUGE/BARTScore metrics
- `llamaindex_quality_metrics.json` - LlamaIndex BLEU/ROUGE/BARTScore metrics

**Other Results:**
- `wandb_export_2025-12-15T21_16_49.900-05_00.csv` - Complete results export (if using WandB)
- `figures/` - Generated visualization figures

### Key Findings

- **LangChain**: Lower latency (4.8-4.9s) but lower retrieval recall (0.31-0.34)
- **LlamaIndex**: Higher latency (8.5-30.9s) but superior retrieval recall (0.52-0.80) and tool efficiency (0.94-0.96)

### Expected Performance Metrics

**LangChain Tool-Use:**
- F1 Score: 0.317-0.335
- Retrieval Recall@k: 0.309-0.342
- Avg Latency: 4.84-4.89s
- Tool Efficiency: 0.841-0.876

**LlamaIndex Tool-Use:**
- F1 Score: 0.319-0.341
- Retrieval Recall@k: 0.516-0.795
- Avg Latency: 8.49-30.88s
- Tool Efficiency: 0.935-0.957

### Runtime Estimates

- Vector store building: ~2-4 hours (depends on dataset size)
- Simple RAG evaluation: ~1-2 hours per framework
- Tool-use evaluation: ~5-10 hours per framework per k value
- Total: ~30-50 hours for complete reproduction

## Reproduction Guide

To reproduce the experimental results from the paper:

1. Follow the **Installation & Setup** section above
2. Complete **Data Setup** with the CUAD dataset
3. Run all evaluations as described in **Usage** section
4. Evaluate response quality metrics using `evaluate_response_quality.py` (optional)
5. Export results from WandB (or use provided export)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the `src/` directory or add it to PYTHONPATH
2. **API Key Errors**: Verify your OpenAI API key is set correctly
3. **Memory Issues**: Reduce batch sizes or use smaller test sets
4. **Vector Store Not Found**: Run the RAG pipeline scripts first to create vector stores
5. **Out of Memory**: Reduce batch sizes or use smaller test sets
6. **API Rate Limits**: Add delays between API calls
7. **Vector Store Errors**: Rebuild vector stores if corrupted
8. **Figure Generation Errors**: Ensure CSV file path is correct


### Questions?

Feel free to open an issue for any questions or concerns.

