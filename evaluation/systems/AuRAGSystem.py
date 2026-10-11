"""
AuRAG System Wrapper for Template 2 Evaluation
==============================================

Wraps the AuRAGPipeline (SPHR + RDG) for end-to-end RAG evaluation.
This system uses grammar-constrained generation to enforce hard citation constraints.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root and src/ so imports resolve regardless of cwd
_ROOT = Path(__file__).parent.parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from RDG import get_rdg_pipeline, DEFAULT_N_CTX as RDG_DEFAULT_N_CTX

# Retrieval: re-use EXACTLY the same index builder as evaluate_retrieval.py (system=sphr),
# so the end-to-end SPHR (incl. cross-reference enrichment) is the one reported in Layer 1.
from evaluation.evaluate_retrieval import build_sphr_index

from .base import RAGSystem


DEFAULT_N_CTX = RDG_DEFAULT_N_CTX


class AuRAGSystem(RAGSystem):
    """
    AuRAG system (SPHR + RDG) for Template 2 evaluation.
    
    Architecture:
    - Layer 1: SPHR hierarchical retrieval with parent-child structure
    - Layer 2: RDG grammar-constrained generation with hard citation constraints
    
    This implementation differs from src/AuRAG.py in that it:
    1. Accepts a pre-built corpus (no document loading)
    2. Returns timing information for evaluation metrics
    3. Extracts article IDs from citations for hallucination measurement
    """
    
    def __init__(self, system_name: str = "aurag"):
        self.system_name = system_name
        self.corpus = None
        self.vectordb = None
        self.retriever = None
        self.rdg = None
        self.embedding_model = None
        
        # Track statistics
        self.retrieval_times = []
        self.generation_times = []
        
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        llm_model_path: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """
        Initialize AuRAG system with corpus.
        
        Args:
            corpus: Dict mapping article_id -> {'caption': article_number, 'text': full_text}
            embedding_model: Embedding model name (e.g., 'intfloat/multilingual-e5-base')
            llm_model_path: Path to GGUF model file for RDG
            db_path: Path for ChromaDB persistence
            **kwargs: Additional config (child_chunk_size, top_k, n_ctx, etc.)
        """
        print(f"\n{'='*70}")
        print(f"Setting up {self.system_name.upper()} System (SPHR + RDG)")
        print(f"{'='*70}")
        
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.llm_model_path = llm_model_path
        
        # Extract configuration
        child_chunk_size = kwargs.get('child_chunk_size', 300)
        child_chunk_overlap = kwargs.get('child_chunk_overlap', 90)
        self.top_k = kwargs.get('top_k', 5)
        self.context_budget = kwargs.get('context_budget', 12000)
        self.retrieval_mode = kwargs.get('retrieval_mode', 'dense')
        self.bm25_weight = kwargs.get('bm25_weight', 0.5)
        self.rrf_k = kwargs.get('rrf_k', 60)
        rebuild_index = kwargs.get('rebuild_index', False)
        n_ctx = kwargs.get('n_ctx', DEFAULT_N_CTX)
        n_gpu_layers = kwargs.get('n_gpu_layers', -1)
        max_tokens = kwargs.get('max_tokens', None)
        seed = kwargs.get('seed', 42)
        
        print(f"\n📋 Configuration:")
        print(f"  - Child chunk size: {child_chunk_size}")
        print(f"  - Child chunk overlap: {child_chunk_overlap}")
        print(f"  - Top-K retrieval: {self.top_k}")
        print(f"  - Context budget: {self.context_budget} tokens")
        print(f"  - Retrieval mode: {self.retrieval_mode}")
        if self.retrieval_mode == 'hybrid':
            print(f"  - BM25 weight: {self.bm25_weight}")
            print(f"  - RRF-k: {self.rrf_k}")
        print(f"  - Model context: {n_ctx} tokens")
        print(f"  - Generation max tokens: {'auto' if max_tokens is None else max_tokens}")
        print(f"  - GPU layers: {n_gpu_layers}")
        print(f"  - Seed: {seed}")
        print(f"  - Embedding device: {kwargs.get('device', 'cpu')}")
        print(f"  - Rebuild index: {rebuild_index}")
        
        # Steps 1-3: SPHR chunking, cross-reference enrichment, vector index and retriever.
        # Calls the IDENTICAL build_sphr_index() used in evaluate_retrieval.py.
        print(f"\n🔹 Steps 1-3: SPHR Index (same builder as Layer-1 evaluation)")
        self.retriever = build_sphr_index(
            corpus=corpus,
            embedding_model=embedding_model,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            top_k=self.top_k,
            context_budget=self.context_budget,
            persist_dir=Path(db_path),
            device=kwargs.get('device', 'cpu'),
            normalize_embeddings=kwargs.get('normalize_embeddings', True),
            rebuild_index=rebuild_index,
            retrieval_mode=self.retrieval_mode,
            bm25_weight=self.bm25_weight,
            rrf_k=self.rrf_k,
        )
        self.vectordb = self.retriever.vectordb
        print(f"  ✓ Retriever ready")
        
        # Step 4: Initialize RDG pipeline (lazy loading for efficiency)
        print(f"\n🔹 Step 4: RDG Pipeline Ready (will load on first query)")
        print(f"  - Model: {llm_model_path}")
        self.rdg_config = {
            'model_path': llm_model_path,
            'n_ctx': n_ctx,
            'n_gpu_layers': n_gpu_layers,
            'max_tokens': max_tokens,
            'seed': seed,
        }
        
        print(f"\n{'='*70}")
        print(f"✅ {self.system_name.upper()} System Ready")
        print(f"{'='*70}\n")
    
    def _get_rdg(self):
        """Lazy load RDG pipeline on first query"""
        if self.rdg is None:
            print(f"\n🔧 Loading RDG model (first query)...")
            self.rdg = get_rdg_pipeline(
                model_path=self.rdg_config['model_path'],
                n_ctx=self.rdg_config['n_ctx'],
                n_gpu_layers=self.rdg_config['n_gpu_layers'],
                seed=self.rdg_config['seed'],
            )
            print(f"  ✓ Model loaded")
        return self.rdg
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process end-to-end RAG query: retrieve + generate with citations.
        
        Args:
            question: User question
            top_k: Override retrieval top-k (optional)
            
        Returns:
            {
                'answer': str,
                'reasoning': str,  # Structured CoT JSON string
                'citations': List[str],  # Article IDs cited in answer
                'retrieved': List[str],  # Article IDs retrieved
                'retrieval_time': float,
                'generation_time': float
            }
        """
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Retrieve context
        t0 = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - t0
        self.retrieval_times.append(retrieval_time)
        
        # Extract article IDs from retrieved documents
        # IMPORTANT: These are the "retrieved" articles used in hallucination check
        # Each doc.metadata['contract_id'] contains the article number (e.g., "123", "456")
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
                'retrieval_time': retrieval_time,
                'generation_time': 0.0
            }
        
        # Step 2: Generate with RDG (model load on the first query is kept out of generation_time)
        rdg = self._get_rdg()
        t1 = time.time()
        try:
            structured_output = rdg.generate(
                question=question,
                documents=docs,
                max_tokens=self.rdg_config.get('max_tokens'),
                answer_mode="yn",
            )
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)
            json_parse_success = 1
            if structured_output.metadata and 'json_parse_success' in structured_output.metadata:
                json_parse_success = int(structured_output.metadata['json_parse_success'])
            
            # Use collector-derived citations from RDG output as the single source of truth.
            cited_article_ids = sorted(set(
                c.contract_id for c in structured_output.citations if c.contract_id and c.contract_id != "Unknown"
            ))
            
            return {
                'answer': structured_output.answer,
                'reasoning': structured_output.to_dict(),  # Changed from to_json() to to_dict()
                'citations': cited_article_ids,
                'retrieved': retrieved_article_ids,
                'json_parse_success': json_parse_success,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
            
        except Exception as e:
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)
            print(f"⚠️  RDG generation failed: {e}")
            
            return {
                'answer': f"Error: {str(e)}",
                'reasoning': "",
                'citations': [],
                'retrieved': retrieved_article_ids,
                'json_parse_success': 0,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return system metadata for reporting"""
        return {
            'system_name': self.system_name,
            'architecture': 'SPHR + RDG (Grammar-Constrained)',
            'embedding_model': self.embedding_model,
            'llm_model': self.rdg_config.get('model_path', 'unknown'),
            'top_k': self.top_k,
            'context_budget': self.context_budget,
            'retrieval_mode': self.retrieval_mode,
            'bm25_weight': self.bm25_weight,
            'rrf_k': self.rrf_k,
            'num_queries_processed': len(self.retrieval_times),
            'avg_retrieval_time': sum(self.retrieval_times) / len(self.retrieval_times) if self.retrieval_times else 0,
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.rdg:
            del self.rdg
            self.rdg = None
        if self.vectordb:
            self.vectordb = None
