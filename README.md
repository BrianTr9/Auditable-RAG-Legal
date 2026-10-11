# Auditable RAG for Legal Documents

Retrieval-augmented generation for statutory law and contracts, built so that every cited source can be checked.

This repository implements a two-layer RAG pipeline for high-stakes legal question answering. The first layer retrieves whole legal provisions (articles, sections) with stable source IDs. The second layer constrains decoding so the model can only cite identifiers that were actually retrieved.

![Status](https://img.shields.io/badge/Status-Research%20Preview-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Motivation

RAG systems used on legal text can fail in four ways when they cite sources:

1. **Type 1 (Fabrication):** citing a source that was never retrieved.
2. **Type 2 (Misattribution):** citing a retrieved source that is irrelevant to the claim.
3. **Type 3 (Reasoning error):** citing a relevant source but drawing a flawed conclusion.
4. **Type 4 (Retrieval failure):** the necessary evidence is missing from the retrieved set.

A fabricated identifier is the worst case for auditing, because there is no document to check. Prompting alone does not rule it out.

## Approach

### Layer 1: provision-aligned hierarchical retrieval
*Targets Type 4.*

- Splits each document at its own provision boundaries (Japanese articles, `Section N`, `Article N`, numbered clauses) and keeps each provision as a parent unit with a stable ID.
- Embeds small child spans (about 300 tokens, split at sentence and clause boundaries) for precise matching, then returns the full parent provision to the LLM.
- Links inline cross-references (for example "see Section 12" or "第七条") to the referenced provision, so related provisions can be returned together.
- Supports dense, BM25, and hybrid ranking (weighted reciprocal rank fusion), with a token budget on the returned context.

### Layer 2: citation-constrained decoding
*Eliminates Type 1 for grammar-accepted outputs and reduces Types 2 and 3.*

- Builds a GBNF grammar for each query from the identifiers that were retrieved, so the citation fields can only contain those identifiers.
- Forces a structured reasoning format: each step holds an extracted snippet, a deduction, a step type (`premise`, `inference`, `conclusion`), and its citations, followed by the answer.
- Generation runs locally with `llama-cpp-python`; the main pipeline makes no API calls.

The guarantee is about membership only: a cited identifier always belongs to the retrieved set. It does not mean the cited provision is relevant, current, or legally correct.

## Technical stack

| Component | Choice |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` (CLI default), `intfloat/multilingual-e5-small` (benchmark evaluation) |
| Sparse retrieval | `rank-bm25` (Japanese handled with character bigrams) |
| LLM inference | `llama-cpp-python`, `Meta-Llama-3.1-8B-Instruct` (GGUF, Q4_K_M) |
| Constraint | Custom GBNF grammar generated per query |

## Quick start

### Prerequisites
- Python 3.10+
- `make`, `cmake` (for the llama.cpp build)

### One-step setup (recommended)

The setup script creates the environment, downloads the GGUF model, and prepares the data folders.

```bash
git clone https://github.com/BrianTr9/Auditable-RAG-Legal.git
cd Auditable-RAG-Legal
chmod +x setup.sh
./setup.sh
```

### Manual installation

1. **Environment and dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Models**
   - LLM: download `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (or similar) into `models/`.
   - Embeddings: `BAAI/bge-small-en-v1.5` is downloaded automatically by HuggingFace.

3. **Data**
   - Create `data/raw/` and place your legal documents (`.pdf` or `.txt`) there.
   - `data/processed/` and `data/vector_db/` are created when you run the program.

### Running the pipeline
```bash
python src/AuRAG.py
```

The program will:
1. Parse and chunk your documents hierarchically.
2. Rebuild the local vector database (Chroma).
3. Start an interactive CLI for questions.

## Evaluation

The evaluation harness covers two benchmarks and reports retrieval and generation separately, so each layer can be tested on its own.

- **COLIEE Task 3 (statute law):** Japanese Civil Code, yes/no entailment questions with gold article IDs.
- **CUAD (LegalBench-RAG):** long contracts with snippet-level ground truth.

### Metrics
- **Fabricated-citation rate:** share of cited IDs that are not in the retrieved set (zero by construction for grammar-accepted outputs).
- **Citation precision and recall:** cited IDs against gold IDs, which reflects Type 2 and 3 errors.
- **Retrieval Hit@K, Recall@K, MRR, NDCG@K:** Layer 1 quality. For CUAD, snippet recall, context tokens, and evidence density.
- **Answer accuracy, JSON compliance, latency.**

### Run
```bash
# Minimal CLI (quick run)
python3 evaluation/evaluate_end2end.py --year R06 --system aurag

# Complete CLI (reproducible, model-agnostic token budget)
python3 evaluation/evaluate_end2end.py \
    --corpus benchmark/COLIEE/civil.xml \
    --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
    --year R06 \
    --system aurag \
    --embedding multilingual \
    --llm-model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --n-ctx 16384 \
    --max-tokens 2048 \
    --top-k 5 \
    --retrieval-mode hybrid \
    --bm25-weight 0.5 \
    --rrf-k 60 \
    --rebuild-index \
    --seed 42

# Generation only, with gold articles as context (isolates Layer 2)
python3 evaluation/evaluate_generation.py --mode rdg --llm-model <model.gguf> --n-ctx <ctx> --max-tokens 2048

# Retrieval only (isolates Layer 1)
python3 evaluation/evaluate_retrieval.py --system sphr --top-k "3,5,10" --retrieval-mode hybrid
```

Baselines available in the harness: fixed-window flat chunking, prompt-only JSON generation (no grammar), and GPT-4o (set `OPENAI_API_KEY`).

## Limitations

- The constraint applies to the citation fields. Free text elsewhere in the output is not constrained.
- Constrained decoding is slower than prompt-only decoding in this implementation; grammar compilation and caching are not optimized.
- Version selection (amendments, repealed provisions, jurisdictions) is outside the current retrieval and grammar.
- Cross-reference linking is pattern-based, so unusual citation styles may be missed.

## Project structure
```
.
├── src/
│   ├── AuRAG.py       # Pipeline entry point and CLI
│   ├── SPHR.py        # Layer 1: hierarchical chunking and retrieval
│   └── RDG.py         # Layer 2: grammar-constrained generation
├── evaluation/
│   ├── evaluate_end2end.py     # Retrieval + generation benchmark
│   ├── evaluate_generation.py  # Layer 2 evaluation with gold context
│   ├── evaluate_retrieval.py   # Layer 1 evaluation
│   ├── adapters/               # Dataset loaders (COLIEE, CUAD, HousingQA)
│   └── systems/                # System wrappers used by the benchmark
├── benchmark/COLIEE/  # Civil Code corpus and query sets
├── data/
│   ├── raw/           # Input documents
│   └── vector_db/     # Persistent ChromaDB
└── models/            # GGUF models
```

## License
MIT License. See [LICENSE](LICENSE) for details.

## Author
**Trung Bao (Brian) Truong**  
*Honours Thesis Project*
