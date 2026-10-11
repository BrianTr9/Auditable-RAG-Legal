#!/usr/bin/env python3
"""
Template 1: Retrieval-Only Evaluation (Layer 1)
================================================

Evaluates retrieval quality only (no generation) on COLIEE dataset.
Metrics: Recall@K, Precision@K, F1@K, Hit@K, MRR, NDCG@K.

Systems:
- sphr: Structure-Preserving Hierarchical Retrieval (AuRAG Layer 1)
- flat: Flat chunking baseline (fixed-size, no hierarchy)

Usage:
    Minimal CLI (quick run):
    python3 evaluation/evaluate_retrieval.py --system sphr --top-k 5 --dataset coliee

    Complete CLI (reproducible benchmark):
    python3 evaluation/evaluate_retrieval.py \
        --dataset coliee \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --system sphr \
        --top-k "3,5,10" \
        --retrieval-mode hybrid \
        --bm25-weight 0.5 \
        --rrf-k 60 \
        --chunk-workers 3 \
        --chunk-map-chunksize 16 \
        --rebuild-index \
        --seed 42

    Baseline comparison (same retrieval settings):
    python3 evaluation/evaluate_retrieval.py --system sphr --retrieval-mode hybrid --top-k 5
    python3 evaluation/evaluate_retrieval.py --system flat --retrieval-mode hybrid --top-k 5

    Smoke test (10 queries):
    python3 evaluation/evaluate_retrieval.py --system sphr --top-k 5 --sample-queries 10
"""

import argparse
import json
import random
import shutil
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, stdev

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# Add src/ and project root to path for local imports
EVAL_PATH = Path(__file__).parent
ROOT_PATH = EVAL_PATH.parent
sys.path.insert(0, str(ROOT_PATH))

SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from SPHR import HierarchicalChunker, HierarchicalRetriever
from evaluation.adapters import available_datasets, load_dataset
from evaluation.dataset_defaults import resolve_dataset_paths

try:
    import tiktoken
except Exception:
    tiktoken = None


def chunk_flat(text: str, contract_id: str, chunk_size: int = 300, overlap: int = 90) -> List[Document]:
    """Flat chunking baseline aligned to token budget when possible."""
    if not text:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    chunk_idx = 0

    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), step):
            token_slice = tokens[i:i + chunk_size]
            if not token_slice:
                continue
            chunk_text = encoding.decode(token_slice)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    'chunk_type': 'flat',
                    'contract_id': contract_id,
                    'chunk_id': f"{contract_id}_flat_{chunk_idx}",
                    'token_count': len(token_slice)
                }
            ))
            chunk_idx += 1
        return chunks

    # Fallback: word-based chunking if tokenizer is unavailable
    words = text.split()
    if not words:
        return []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                'chunk_type': 'flat',
                'contract_id': contract_id,
                'chunk_id': f"{contract_id}_flat_{chunk_idx}",
                'token_count': len(chunk_words)
            }
        ))
        chunk_idx += 1
    return chunks


class FlatRetriever:
    """Flat retriever with dense/bm25/hybrid ranking on flat chunks."""

    def __init__(
        self,
        vectordb,
        flat_documents: List[Document],
        top_k: int = 5,
        retrieval_mode: str = 'dense',
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vectordb = vectordb
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode.lower()
        self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        self.dense_weight = 1.0 - self.bm25_weight
        self.rrf_k = max(1, int(rrf_k))

        if self.retrieval_mode not in {'dense', 'hybrid'}:
            raise ValueError(
                f"Invalid retrieval_mode='{retrieval_mode}'. Choose from: dense, hybrid"
            )

        self.chunk_id_to_doc: Dict[str, Document] = {}
        self.chunk_ids: List[str] = []
        for doc in flat_documents:
            chunk_id = doc.metadata.get('chunk_id')
            if not chunk_id:
                continue
            if chunk_id not in self.chunk_id_to_doc:
                self.chunk_id_to_doc[chunk_id] = doc
                self.chunk_ids.append(chunk_id)

        tokenized = [self._bm25_tokenize(self.chunk_id_to_doc[cid].page_content) for cid in self.chunk_ids]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        if not text:
            return []

        s = str(text).lower()
        tokens: List[str] = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s)

        cjk_spans = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", s)
        for span in cjk_spans:
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[i:i+2] for i in range(len(span) - 1))

        return tokens

    def _dense_chunk_ids(self, query: str) -> List[str]:
        docs = self.vectordb.similarity_search(query, k=self.top_k)
        seen = set()
        ranked = []
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ranked.append(chunk_id)
        return ranked[:self.top_k]

    def _bm25_chunk_ids(self, query: str) -> List[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._bm25_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return [self.chunk_ids[idx] for idx in ranked_indices[:self.top_k]]

    def _rank_chunk_ids(self, query: str) -> List[str]:
        if self.retrieval_mode == 'dense':
            return self._dense_chunk_ids(query)

        dense_ids = self._dense_chunk_ids(query)
        bm25_ids = self._bm25_chunk_ids(query)

        if not bm25_ids:
            return dense_ids

        scores: Dict[str, float] = {}
        for rank, chunk_id in enumerate(dense_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self.dense_weight / (self.rrf_k + rank)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:self.top_k]]

    def invoke(self, query: str) -> List[Document]:
        ranked_ids = self._rank_chunk_ids(query)
        return [self.chunk_id_to_doc[cid] for cid in ranked_ids if cid in self.chunk_id_to_doc]


def _chunk_one_contract_worker(task):
    article_id, article_text, local_child_size, local_child_overlap = task
    local_chunker = HierarchicalChunker(child_size=local_child_size, child_overlap=local_child_overlap)
    unit_id = str(article_id)
    parent_chunks, child_chunks = local_chunker.chunk_contract(article_text, contract_id=unit_id)

    local_parent_to_children = {}
    for parent in parent_chunks:
        parent_id = parent.chunk_id
        local_parent_to_children[parent_id] = [
            child.chunk_id for child in child_chunks
            if child.parent_id == parent_id
        ]

    return parent_chunks, child_chunks, local_parent_to_children


def build_sphr_index(
    corpus: Dict[str, Dict[str, str]],
    embedding_model: str,
    child_chunk_size: int,
    child_chunk_overlap: int,
    top_k: int,
    context_budget: int,
    persist_dir: Path,
    device: str,
    normalize_embeddings: bool,
    rebuild_index: bool,
    retrieval_mode: str,
    bm25_weight: float,
    rrf_k: int,
    chunk_workers: int = 1,
    chunk_map_chunksize: int = 16,
):
    """Build SPHR index and retriever."""

    def chunk_contract_sequential(chunker: HierarchicalChunker):
        all_parent_chunks_local = []
        all_child_chunks_local = []
        parent_to_children_local = {}

        for article_id, article_data in corpus.items():
            article_text = article_data['text']
            # IMPORTANT: use canonical corpus unit_id as retrieval identity.
            # `caption` is display metadata and may not be unique/stable across datasets.
            unit_id = str(article_id)
            parent_chunks, child_chunks = chunker.chunk_contract(article_text, contract_id=unit_id)
            all_parent_chunks_local.extend(parent_chunks)
            all_child_chunks_local.extend(child_chunks)
            for parent in parent_chunks:
                parent_id = parent.chunk_id
                parent_to_children_local[parent_id] = [
                    child.chunk_id for child in child_chunks
                    if child.parent_id == parent_id
                ]

        return all_parent_chunks_local, all_child_chunks_local, parent_to_children_local

    def chunk_contract_parallel(chunker: HierarchicalChunker):
        all_parent_chunks_local = []
        all_child_chunks_local = []
        parent_to_children_local = {}

        tasks = (
            (article_id, article_data['text'], child_chunk_size, child_chunk_overlap)
            for article_id, article_data in corpus.items()
        )

        print(f"🚀 Parallel chunking enabled: workers={chunk_workers}, map_chunksize={chunk_map_chunksize}")
        with ProcessPoolExecutor(max_workers=chunk_workers) as executor:
            for parent_chunks, child_chunks, local_parent_to_children in executor.map(
                _chunk_one_contract_worker,
                tasks,
                chunksize=chunk_map_chunksize,
            ):
                all_parent_chunks_local.extend(parent_chunks)
                all_child_chunks_local.extend(child_chunks)
                parent_to_children_local.update(local_parent_to_children)

        return all_parent_chunks_local, all_child_chunks_local, parent_to_children_local

    chunker = HierarchicalChunker(child_size=child_chunk_size, child_overlap=child_chunk_overlap)
    if chunk_workers and chunk_workers > 1:
        all_parent_chunks, all_child_chunks, parent_to_children = chunk_contract_parallel(chunker)
    else:
        all_parent_chunks, all_child_chunks, parent_to_children = chunk_contract_sequential(chunker)

    # Enrich parent-child graph with cross-references (SPHR intent)
    parent_to_children = chunker.enrich_parent_to_children_with_cross_refs(
        parent_to_children,
        all_parent_chunks,
        all_child_chunks
    )

    splits = [
        Document(
            page_content=child.text,
            metadata={
                'chunk_id': child.chunk_id,
                'chunk_type': 'child',
                'parent_id': child.parent_id,
                'section_num': child.section_num,
                'section_title': child.section_title,
                'contract_id': child.contract_id,
                'token_count': child.token_count
            }
        )
        for child in all_child_chunks
    ]

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )

    if rebuild_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(persist_dir)
    )

    retriever = HierarchicalRetriever(
        vectordb,
        all_parent_chunks,
        parent_to_children,
        top_k=top_k,
        context_budget=context_budget,
        retrieval_mode=retrieval_mode,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )

    return retriever


def build_flat_index(
    corpus: Dict[str, Dict[str, str]],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    retrieval_mode: str,
    bm25_weight: float,
    rrf_k: int,
    persist_dir: Path,
    device: str,
    normalize_embeddings: bool,
    rebuild_index: bool
):
    """Build flat baseline index."""
    documents = []
    for article_id, article_data in corpus.items():
        article_text = article_data['text']
        unit_id = str(article_id)
        documents.extend(chunk_flat(article_text, contract_id=unit_id, chunk_size=chunk_size, overlap=overlap))

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )

    if rebuild_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(persist_dir)
    )

    return FlatRetriever(
        vectordb=vectordb,
        flat_documents=documents,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )


def calculate_retrieval_metrics(retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    overlap = retrieved_set & relevant_set

    hit = 1.0 if overlap else 0.0
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(relevant_set) if relevant_set else 0.0

    # F1@K
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # MRR: rank of first relevant
    mrr = 0.0
    if relevant_set:
        for idx, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                mrr = 1.0 / (idx + 1)
                break

    # NDCG@K for binary relevance
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(relevant_set), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        'hit@k': hit,
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'mrr': mrr,
        'ndcg@k': ndcg,
        'num_retrieved': len(retrieved_set),
        'num_gt': len(relevant_set),
        'num_overlap': len(overlap)
    }


def calculate_snippet_retrieval_metrics(retrieved_docs: List[Document], gt_snippets: List[str]) -> Dict[str, float]:
    """
    Evaluates snippet-level retrieval for datasets like CUAD.
    - A ground truth snippet is 'found' if it appears in any retrieved document's text.
    - Recall: fraction of gt_snippets found.
    - Precision: fraction of retrieved_docs that contain at least one gt_snippet.
    - MRR: Highest rank (idx + 1) among retrieved_docs that contains at least one gt_snippet.
    - NDCG: binary gain per retrieved doc, counted only when the doc contains at least one
      gt_snippet that no higher-ranked doc already matched (novelty-aware gain). Without this,
      overlapping chunks that contain the same snippet are counted repeatedly while the ideal
      DCG counts each snippet once, which lets NDCG exceed 1. With it, DCG <= IDCG always,
      so NDCG is in [0, 1]. IDCG keeps the documented normalisation min(|GT|, |returned|).
    """
    relevant_snippets_total = len(gt_snippets)
    
    context_tokens = 0
    encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None
    for doc in retrieved_docs:
        if 'token_count' in doc.metadata:
            context_tokens += doc.metadata['token_count']
        elif encoding:
            context_tokens += len(encoding.encode(doc.page_content))
        else:
            context_tokens += len(doc.page_content.split())
            
    if relevant_snippets_total == 0:
        return {
            'hit@k': 0.0, 'precision@k': 0.0, 'recall@k': 0.0, 'f1@k': 0.0, 'mrr': 0.0, 'ndcg@k': 0.0,
            'num_retrieved': len(retrieved_docs), 'num_gt': relevant_snippets_total, 'num_overlap': 0,
            'context_tokens': context_tokens,
            'evidence_density': 0.0
        }
        
    found_snippets = set()
    relevant_doc_indices = []
    novel_doc_indices = []  # docs that add >=1 snippet not matched at a higher rank (NDCG gain)
    found_snippet_tokens = 0
    
    for i, doc in enumerate(retrieved_docs):
        doc_is_relevant = False
        doc_adds_new_snippet = False
        for s_idx, snippet_dict in enumerate(gt_snippets):
            # snippet_dict contains 'file_path', 'span', 'answer'
            # The actual snippet text might be in 'span' or 'answer' depending on CUAD
            # CUAD 'answer' usually contains the extracted text.
            snippet_text = snippet_dict.get('answer', '')
            if not snippet_text:
                continue
                
            # If the snippet text was found inside this retrieved document chunk:
            if snippet_text in doc.page_content:
                if s_idx not in found_snippets:
                    found_snippets.add(s_idx)
                    doc_adds_new_snippet = True
                    if encoding:
                        found_snippet_tokens += len(encoding.encode(snippet_text))
                    else:
                        found_snippet_tokens += len(snippet_text.split())
                doc_is_relevant = True
        
        if doc_is_relevant:
            relevant_doc_indices.append(i)
        if doc_adds_new_snippet:
            novel_doc_indices.append(i)
            
    recall = len(found_snippets) / relevant_snippets_total
    precision = len(relevant_doc_indices) / len(retrieved_docs) if retrieved_docs else 0.0
    
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        
    hit = 1.0 if len(found_snippets) > 0 else 0.0
    mrr = 1.0 / (relevant_doc_indices[0] + 1) if relevant_doc_indices else 0.0
    
    dcg = 0.0
    for idx in novel_doc_indices:
        dcg += 1.0 / math.log2(idx + 2)
    
    ideal_hits = min(relevant_snippets_total, len(retrieved_docs))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    
    evidence_density = found_snippet_tokens / context_tokens if context_tokens > 0 else 0.0
    
    return {
        'hit@k': hit,
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'mrr': mrr,
        'ndcg@k': ndcg,
        'num_retrieved': len(retrieved_docs),
        'num_gt': relevant_snippets_total,
        'num_overlap': len(found_snippets),
        'context_tokens': context_tokens,
        'evidence_density': evidence_density
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieval-Only Evaluation (Layer 1)")
    dataset_choices = available_datasets()
    parser.add_argument('--dataset', type=str, default='coliee', choices=dataset_choices,
                        help='Dataset adapter to use for corpus/query loading')
    parser.add_argument('--corpus', type=str, default=None, help='Optional corpus path override')
    parser.add_argument('--queries', type=str, default=None, help='Optional queries path override')
    parser.add_argument('--system', type=str, default='sphr', choices=['sphr', 'flat'])
    parser.add_argument('--top-k', type=str, default='5', help='Comma-separated list of top-k values, e.g. "3,5,10"')
    parser.add_argument('--embedding', type=str, default='intfloat/multilingual-e5-small')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--no-normalize-embeddings', action='store_false', dest='normalize_embeddings')
    parser.add_argument('--child-chunk-size', type=int, default=300)
    parser.add_argument('--child-chunk-overlap', type=int, default=90)
    parser.add_argument('--retrieval-mode', type=str, default='dense', choices=['dense', 'hybrid'])
    parser.add_argument('--bm25-weight', type=float, default=0.5)
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--chunk-workers', type=int, default=1,
                        help='Number of CPU worker processes for SPHR chunking. Set >1 to enable parallel chunking.')
    parser.add_argument('--chunk-map-chunksize', type=int, default=16,
                        help='Task chunksize for ProcessPoolExecutor.map during parallel chunking.')
    parser.add_argument('--rebuild-index', action='store_true', default=False)
    parser.add_argument('--sample-queries', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    if not hasattr(args, 'normalize_embeddings'):
        args.normalize_embeddings = True
    elif args.normalize_embeddings is None:
        args.normalize_embeddings = True

    random.seed(args.seed)

    workspace_root = Path(__file__).parent.parent
    corpus_path, queries_path = resolve_dataset_paths(
        dataset=args.dataset,
        workspace_root=workspace_root,
        provided_corpus=args.corpus,
        provided_queries=args.queries,
    )
    corpus, queries, dataset_metadata = load_dataset(
        dataset=args.dataset,
        workspace_root=workspace_root,
        corpus_path=corpus_path,
        queries_path=queries_path,
    )

    if args.sample_queries and args.sample_queries > 0:
        # If running a small subset of documents, filter out queries that target missing documents
        if args.dataset == 'cuad' and corpus:
            valid_doc_ids = set(corpus.keys())
            filtered_queries = [
                q for q in queries 
                if any(art in valid_doc_ids for art in q['relevant_articles'])
            ]
            if filtered_queries:
                queries = filtered_queries
                
        queries = random.sample(queries, min(args.sample_queries, len(queries)))

    output_dir = Path(__file__).parent

    # Parse top-k values
    try:
        top_k_list = [int(k.strip()) for k in str(args.top_k).split(',')]
        top_k_list = sorted(list(set(top_k_list)))  # unique and sorted
    except ValueError:
        print(f"❌ Error: Invalid --top-k format: {args.top_k}. Expected comma-separated integers, e.g., '3,5,10'")
        sys.exit(1)

    if not top_k_list:
        top_k_list = [5]

    max_k = max(top_k_list)
    print(f"📊 Running evaluation for top_k values: {top_k_list} (Retrieving max {max_k} documents)")

    # Retrieval-only evaluation: always disable context truncation to isolate ranking quality.
    context_budget = 10**9
    if args.system == 'sphr':
        retriever = build_sphr_index(
            corpus,
            args.embedding,
            args.child_chunk_size,
            args.child_chunk_overlap,
            max_k,
            context_budget,
            output_dir / 'vector_db_sphr',
            args.device,
            args.normalize_embeddings,
            args.rebuild_index,
            args.retrieval_mode,
            args.bm25_weight,
            args.rrf_k,
            args.chunk_workers,
            args.chunk_map_chunksize,
        )
    else:
        retriever = build_flat_index(
            corpus,
            args.embedding,
            args.child_chunk_size,
            args.child_chunk_overlap,
            max_k,
            args.retrieval_mode,
            args.bm25_weight,
            args.rrf_k,
            output_dir / 'vector_db_flat',
            args.device,
            args.normalize_embeddings,
            args.rebuild_index
        )

    per_query_results = []
    hit_scores, prec_scores, rec_scores, f1_scores, mrr_scores, ndcg_scores = [], [], [], [], [], []
    include_cuad_context_metrics = args.dataset == 'cuad'

    def collect_unique_articles(docs: List[Document]) -> List[str]:
        retrieved = []
        for d in docs:
            doc_id = d.metadata.get('contract_id', 'unknown')
            if doc_id not in retrieved:
                retrieved.append(doc_id)
        return retrieved

    print(f"\n📈 Calculating metrics for top_k values: {top_k_list}")
    for current_k in top_k_list:
        print(f"\n🔍 [top_k={current_k}] Retrieving context for {len(queries)} queries...")
        # Update retriever configuration sequentially instead of truncating from max_k
        retriever.top_k = current_k
        
        per_query_results = []
        hit_scores, prec_scores, rec_scores, f1_scores, mrr_scores, ndcg_scores = [], [], [], [], [], []
        context_tokens_scores = []
        evidence_density_scores = []

        for q in queries:
            question = q['question']
            relevant = q['relevant_articles']

            docs = retriever.invoke(question)
            
            if args.dataset == 'cuad':
                gt_snippets = q.get('metadata', {}).get('ground_truth_snippets', [])
                metrics = calculate_snippet_retrieval_metrics(docs, gt_snippets)
                retrieved_k = [doc.metadata.get('contract_id', 'unknown') for doc in docs]
            else:
                retrieved_k = collect_unique_articles(docs)
                metrics = calculate_retrieval_metrics(retrieved_k, relevant)

            hit_scores.append(metrics['hit@k'])
            prec_scores.append(metrics['precision@k'])
            rec_scores.append(metrics['recall@k'])
            f1_scores.append(metrics['f1@k'])
            mrr_scores.append(metrics['mrr'])
            ndcg_scores.append(metrics['ndcg@k'])
            if include_cuad_context_metrics:
                context_tokens_scores.append(metrics['context_tokens'])
                evidence_density_scores.append(metrics['evidence_density'])
            
            per_query_results.append({
                'query_id': q['query_id'],
                'ground_truth': relevant,
                'retrieved': retrieved_k,
                'metrics': metrics
            })

        aggregate = {
            'hit@k': {'mean': mean(hit_scores), 'std': stdev(hit_scores) if len(hit_scores) > 1 else 0.0},
            'precision@k': {'mean': mean(prec_scores), 'std': stdev(prec_scores) if len(prec_scores) > 1 else 0.0},
            'recall@k': {'mean': mean(rec_scores), 'std': stdev(rec_scores) if len(rec_scores) > 1 else 0.0},
            'f1@k': {'mean': mean(f1_scores), 'std': stdev(f1_scores) if len(f1_scores) > 1 else 0.0},
            'mrr': {'mean': mean(mrr_scores), 'std': stdev(mrr_scores) if len(mrr_scores) > 1 else 0.0},
            'ndcg@k': {'mean': mean(ndcg_scores), 'std': stdev(ndcg_scores) if len(ndcg_scores) > 1 else 0.0},
        }

        if include_cuad_context_metrics:
            aggregate['context_tokens'] = {
                'mean': mean(context_tokens_scores),
                'std': stdev(context_tokens_scores) if len(context_tokens_scores) > 1 else 0.0,
            }
            aggregate['evidence_density'] = {
                'mean': mean(evidence_density_scores) if evidence_density_scores else 0.0,
                'std': stdev(evidence_density_scores) if len(evidence_density_scores) > 1 else 0.0,
            }

        output_metadata = {
            'dataset': args.dataset,
            **dataset_metadata,
            'system': args.system,
            'embedding': args.embedding,
            'top_k': current_k,
            'num_queries': len(per_query_results),
            'child_chunk_size': args.child_chunk_size,
            'child_chunk_overlap': args.child_chunk_overlap,
            'context_budget': context_budget,
            'retrieval_mode': args.retrieval_mode,
            'bm25_weight': args.bm25_weight,
            'rrf_k': args.rrf_k,
            'chunk_workers': args.chunk_workers,
            'chunk_map_chunksize': args.chunk_map_chunksize,
            'device': args.device,
            'normalize_embeddings': args.normalize_embeddings,
            'rebuild_index': args.rebuild_index,
            'seed': args.seed
        }

        output = {
            'metadata': output_metadata,
            'aggregate_metrics': aggregate,
            'per_query_results': per_query_results
        }

        if args.output:
            # If using custom output, suffix the K value before the extension
            base_path = Path(args.output)
            output_path = base_path.with_name(f"{base_path.stem}_k{current_k}{base_path.suffix}")
        else:
            output_path = output_dir / f"evaluation_results_layer1_retrieval_{args.system}_k{current_k}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved top_k={current_k}: \thit@k={aggregate['hit@k']['mean']:.4f}\trecall@k={aggregate['recall@k']['mean']:.4f}\t-> {output_path.name}")

    print(f"\n🎉 All {len(top_k_list)} multi-top_k evaluations complete!")


if __name__ == '__main__':
    main()
