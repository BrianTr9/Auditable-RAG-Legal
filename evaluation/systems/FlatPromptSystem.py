"""
FlatPrompt Baseline System for End-to-End Evaluation
=====================================================

Baseline RAG system that combines:
  - Flat fixed-size token chunking + retrieval  →  same code as evaluate_retrieval.py (system=flat)
  - Prompt-only JSON generation (no grammar)    →  same code as evaluate_generation.py (mode=prompt)

TRUTHFULNESS GUARANTEE
----------------------
This class never re-implements chunking or generation logic.
It imports and calls the *exact same* functions used by the two isolated-layer evaluations:

    Layer 1 (evaluate_retrieval.py)  :  build_flat_index()
                                        (internally uses chunk_flat() + FlatRetriever)
    Layer 2 (evaluate_generation.py) :  prompt_only_generate()
                                        extract_article_ids_from_citations()

Any change to those functions automatically propagates here, keeping the E2E baseline
numerically consistent with the per-layer ablations.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

# Add project root and src/ so imports resolve regardless of cwd
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from RDG import get_rdg_pipeline, DEFAULT_N_CTX as RDG_DEFAULT_N_CTX

# ── Single source of truth imports ────────────────────────────────────────────
# Retrieval: re-use EXACTLY the same helper used in evaluate_retrieval.py
from evaluation.evaluate_retrieval import build_flat_index

# Generation: re-use EXACTLY the same helpers used in evaluate_generation.py
from evaluation.evaluate_generation import (
    prompt_only_generate,
    extract_article_ids_from_citations,  # same citation normalizer as Layer 2 prompt path
)
# ──────────────────────────────────────────────────────────────────────────────

from .base import RAGSystem


DEFAULT_N_CTX = RDG_DEFAULT_N_CTX


class FlatPromptSystem(RAGSystem):
    """
    Flat-chunking + Prompt-only E2E baseline.

    Architecture
    ------------
    Layer 1 — Flat fixed-size chunking (chunk_flat) + dense/hybrid FlatRetriever
    Layer 2 — Prompt-only JSON generation (no grammar-constrained decoding)

    All kwarg names are intentionally kept identical to AuRAGSystem.setup() so
    that evaluate_end2end.py can call either system with the same kwargs dict.
    """

    def __init__(self, system_name: str = "flat_prompt"):
        self.system_name = system_name
        self.retriever = None
        self.rdg = None
        self.embedding_model = None

        self.retrieval_times: List[float] = []
        self.generation_times: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # RAGSystem interface
    # ──────────────────────────────────────────────────────────────────────────

    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        llm_model_path: str,
        db_path: Path,
        **kwargs,
    ) -> None:
        """
        Initialise the FlatPrompt system.

        Retrieval kwargs  →  mirror evaluate_retrieval.py --system flat defaults
        Generation kwargs →  mirror evaluate_generation.py --mode prompt defaults

        Parameters
        ----------
        corpus          : {article_id: {'caption': str, 'text': str}}
        embedding_model : HuggingFace model name (e.g. 'BAAI/bge-small-en-v1.5')
        llm_model_path  : Path to GGUF model file (needed for prompt builder & tokenizer)
        db_path         : ChromaDB persistence directory
        **kwargs        : See table below.

        Accepted kwargs
        ---------------
        child_chunk_size    (int,   default 300)   — flat chunk token size
        child_chunk_overlap (int,   default 90)    — flat chunk overlap in tokens
        top_k               (int,   default 5)     — documents to retrieve
        retrieval_mode      (str,   default dense) — dense | hybrid
        bm25_weight         (float, default 0.5)   — only for hybrid mode
        rrf_k               (int,   default 60)    — only for hybrid mode
        rebuild_index       (bool,  default False)
        device              (str,   default cpu)   — cpu | cuda | mps
        normalize_embeddings(bool,  default True)
        n_ctx               (int,   default RDG default) — LLM context window
        n_gpu_layers        (int,   default -1)    — GPU layers for llama.cpp
        max_tokens          (int|None, default None) — None = auto
        seed                (int,   default 42)
        citation_mode       (str,   default short) — short | both
        """
        print(f"\n{'='*70}")
        print(f"Setting up {self.system_name.upper()} System (Flat Chunking + Prompt-Only)")
        print(f"{'='*70}")

        self.embedding_model = embedding_model
        self.llm_model_path = str(llm_model_path)

        # ── Retrieval config ──────────────────────────────────────────────────
        # Use same kwarg names as AuRAGSystem so evaluate_end2end.py setup() call
        # is identical for both systems.
        chunk_size = int(kwargs.get('child_chunk_size', 300))
        overlap = int(kwargs.get('child_chunk_overlap', 90))
        self.top_k = int(kwargs.get('top_k', 5))
        self.retrieval_mode = str(kwargs.get('retrieval_mode', 'dense'))
        self.bm25_weight = float(kwargs.get('bm25_weight', 0.5))
        self.rrf_k = int(kwargs.get('rrf_k', 60))
        rebuild_index = bool(kwargs.get('rebuild_index', False))
        device = str(kwargs.get('device', 'cpu'))
        normalize_embeddings = bool(kwargs.get('normalize_embeddings', True))

        # ── Generation config ─────────────────────────────────────────────────
        n_ctx = int(kwargs.get('n_ctx', DEFAULT_N_CTX))
        n_gpu_layers = int(kwargs.get('n_gpu_layers', -1))
        self.max_tokens: Optional[int] = kwargs.get('max_tokens', None)
        self.seed = int(kwargs.get('seed', 42))
        self.citation_mode: Literal["short", "both"] = kwargs.get('citation_mode', 'short')

        print(f"\n📋 Configuration:")
        print(f"  - Chunk size / overlap: {chunk_size} / {overlap} tokens")
        print(f"  - Top-K retrieval: {self.top_k}")
        print(f"  - Retrieval mode: {self.retrieval_mode}")
        if self.retrieval_mode == 'hybrid':
            print(f"  - BM25 weight: {self.bm25_weight}, RRF-k: {self.rrf_k}")
        print(f"  - Model context: {n_ctx} tokens")
        print(f"  - Generation max tokens: {'auto' if self.max_tokens is None else self.max_tokens}")
        print(f"  - GPU layers: {n_gpu_layers}")
        print(f"  - Citation mode: {self.citation_mode}")
        print(f"  - Seed: {self.seed}")
        print(f"  - Rebuild index: {rebuild_index}")

        # Step 1 — Build flat index
        # Calls the IDENTICAL build_flat_index() used in evaluate_retrieval.py.
        print(f"\n🔹 Step 1: Flat Chunking + Vector Index")
        self.retriever = build_flat_index(
            corpus=corpus,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=self.top_k,
            retrieval_mode=self.retrieval_mode,
            bm25_weight=self.bm25_weight,
            rrf_k=self.rrf_k,
            persist_dir=Path(db_path),
            device=device,
            normalize_embeddings=normalize_embeddings,
            rebuild_index=rebuild_index,
        )
        print(f"  ✓ FlatRetriever ready")

        # Step 2 — RDG pipeline (lazy load; needed for prompt builder + tokenizer)
        print(f"\n🔹 Step 2: RDG Pipeline (will load on first query)")
        print(f"  - Model: {llm_model_path}")
        self._rdg_init_kwargs = {
            'model_path': str(llm_model_path),
            'n_ctx': n_ctx,
            'n_gpu_layers': n_gpu_layers,
            'seed': self.seed,
        }

        print(f"\n{'='*70}")
        print(f"✅ {self.system_name.upper()} System Ready")
        print(f"{'='*70}\n")

    def _get_rdg(self):
        """Lazy-load RDG pipeline on first query (mirrors AuRAGSystem._get_rdg)."""
        if self.rdg is None:
            print(f"\n🔧 Loading RDG model for prompt builder (first query)...")
            self.rdg = get_rdg_pipeline(**self._rdg_init_kwargs)
            print(f"  ✓ Model loaded")
        return self.rdg

    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        End-to-end query: flat retrieval → prompt-only generation.

        Returns the same dict contract as AuRAGSystem.query() so
        evaluate_end2end.evaluate_rag_system() works without modification.

        Return keys
        -----------
        answer           : str
        reasoning        : dict | str
        citations        : List[str]   article IDs cited in the answer
        retrieved        : List[str]   article IDs returned by the retriever
        json_parse_success : int (0 or 1)
        retrieval_time   : float (seconds)
        generation_time  : float (seconds)
        """
        if top_k is None:
            top_k = self.top_k

        # ── Layer 1: Flat retrieval ───────────────────────────────────────────
        # Uses the IDENTICAL FlatRetriever.invoke() called in evaluate_retrieval.py.
        t0 = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - t0
        self.retrieval_times.append(retrieval_time)

        retrieved_article_ids = sorted(set(
            doc.metadata.get('contract_id', 'unknown')
            for doc in docs
        ))

        if not docs:
            return {
                'answer': "No relevant context found.",
                'reasoning': "",
                'citations': [],
                'retrieved': [],
                'json_parse_success': 0,
                'retrieval_time': retrieval_time,
                'generation_time': 0.0,
            }

        # ── Layer 2: Prompt-only generation ──────────────────────────────────
        # Calls the IDENTICAL prompt_only_generate() used in evaluate_generation.py.
        # Model load on the first query is kept out of generation_time (same as AuRAGSystem).
        rdg = self._get_rdg()
        t1 = time.time()
        try:
            result = prompt_only_generate(
                rdg=rdg,
                question=question,
                documents=docs,
                max_tokens=self.max_tokens,
                temperature=0.0,
                citation_mode=self.citation_mode,
            )
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)

            # Extract article IDs using the IDENTICAL normalizer from evaluate_generation.py
            # (handles contract_id → reference_id fallback and section-suffix stripping).
            raw_citations = result.get('citations', [])
            cited_article_ids = extract_article_ids_from_citations(raw_citations)

            reasoning = result.get('reasoning', "")
            if hasattr(reasoning, 'to_dict'):
                reasoning = reasoning.to_dict()

            return {
                'answer': result.get('answer', ''),
                'reasoning': reasoning,
                'citations': cited_article_ids,
                'retrieved': retrieved_article_ids,
                'json_parse_success': result.get('json_parse_success', 0),
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
            }

        except Exception as exc:
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)
            print(f"⚠️  Prompt generation failed: {exc}")
            return {
                'answer': f"Error: {str(exc)}",
                'reasoning': "",
                'citations': [],
                'retrieved': retrieved_article_ids,
                'json_parse_success': 0,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
            }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'system_name': self.system_name,
            'architecture': 'Flat Chunking + Prompt-Only (No Grammar Constraints)',
            'embedding_model': self.embedding_model,
            'llm_model': self._rdg_init_kwargs.get('model_path', 'unknown'),
            'top_k': self.top_k,
            'retrieval_mode': self.retrieval_mode,
            'bm25_weight': self.bm25_weight,
            'rrf_k': self.rrf_k,
            'citation_mode': self.citation_mode,
            'num_queries_processed': len(self.retrieval_times),
            'avg_retrieval_time': (
                sum(self.retrieval_times) / len(self.retrieval_times)
                if self.retrieval_times else 0.0
            ),
            'avg_generation_time': (
                sum(self.generation_times) / len(self.generation_times)
                if self.generation_times else 0.0
            ),
        }

    def cleanup(self) -> None:
        if self.rdg is not None:
            del self.rdg
            self.rdg = None
        if self.retriever is not None and hasattr(self.retriever, 'vectordb'):
            self.retriever.vectordb = None
