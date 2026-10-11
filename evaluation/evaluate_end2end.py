"""
Template 2: End-to-End RAG Evaluation with Citation Hallucination Measurement
=============================================================================

Evaluates complete RAG systems (retrieval + generation) on:
1. Citation Hallucination Rate: How many cited articles were NOT retrieved?
2. Answer Correctness: Y/N accuracy vs ground truth
3. Latency: Total time per query (retrieval + generation)

This script loads COLIEE entailment pairs (Y/N questions + correct article IDs)
and measures whether RAG systems cite only retrieved articles (hallucination-free).

Usage:
    Minimal CLI (quick run):
    python3 evaluation/evaluate_end2end.py --dataset coliee --year R06 --system aurag

    Complete CLI (reproducible and model-agnostic):
    python3 evaluation/evaluate_end2end.py \
        --dataset coliee \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --year R06 \
        --system aurag \
        --embedding multilingual \
        --llm-model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
        --n-ctx 16384 \
        --max-tokens auto \
        --top-k 5 \
        --retrieval-mode hybrid \
        --bm25-weight 0.5 \
        --rrf-k 60 \
        --rebuild-index \
        --seed 42

    Cross-model ablation pattern (same policy, different model/context):
    python3 evaluation/evaluate_end2end.py --llm-model <model.gguf> --n-ctx <ctx> --max-tokens auto
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from statistics import mean, stdev

# Add project root so `evaluation.*` imports work when running this file directly.
EVAL_PATH = Path(__file__).parent
ROOT_PATH = EVAL_PATH.parent
sys.path.insert(0, str(ROOT_PATH))

from evaluation.systems.base import RAGSystem
from evaluation.systems.AuRAGSystem import DEFAULT_N_CTX
from evaluation.systems.registry import available_systems, create_system
from evaluation.adapters import available_datasets, load_dataset
from evaluation.dataset_defaults import resolve_dataset_paths


def normalize_yn_answer(text: str) -> str | None:
    """Normalize a model answer to a COLIEE-style label: 'Y' or 'N'.

    Since RDG guarantees exact 'Y' or 'N' output, this function is primarily 
    a strict filter that matches 'Y', 'N', 'Yes', 'No' exactly or as complete words.
    
    Returns:
        'Y' if the text strictly means yes,
        'N' if the text strictly means no,
        None if ambiguous.
    """
    if not text:
        return None

    s = text.strip()
    if not s:
        return None
        
    s_upper = s.upper()

    # Exact match (Standard RDG mode)
    if s_upper in ["Y", "N", "YES", "NO"]:
        return s_upper[0]
        
    # Check start of string in case there's punctuation or standard baseline output
    # e.g., "Y.", "N,", "Yes ", "No\n"
    import re
    if re.match(r'^(Y|YES)[^a-zA-Z0-9]', s_upper):
        return "Y"
    if re.match(r'^(N|NO)[^a-zA-Z0-9]', s_upper):
        return "N"

    # Allow matches like "Answer: Y"
    s_lower = s.lower()
    for p in ("answer:", "label:", "prediction:", "decision:"):
        if p in s_lower:
            idx = s_lower.rfind(p) + len(p)
            remainder = s_lower[idx:].strip().upper()
            if remainder in ("Y", "YES") or re.match(r'^(Y|YES)[^a-zA-Z0-9]', remainder):
                return "Y"
            if remainder in ("N", "NO") or re.match(r'^(N|NO)[^a-zA-Z0-9]', remainder):
                return "N"

    # Common exact mappings for Japanese baselines just in case
    if s == "はい" or s.startswith("はい"): return "Y"
    if s == "いいえ" or s.startswith("いいえ"): return "N"

    return None


def calculate_citation_metrics(
    citations: List[str],
    retrieved: List[str],
    relevant_articles: List[str]
) -> Dict[str, float]:
    """
    Calculate citation hallucination and retrieval metrics.
    
    CRITICAL: Hallucination is SYNTACTIC check (cited but NOT retrieved),
    not semantic check (cited but not relevant).
    
    Args:
        citations: Article IDs cited in the generated answer (from LLM output)
        retrieved: Article IDs retrieved by the system (from retrieval step)
        relevant_articles: Ground truth relevant article IDs (from COLIEE labels)
        
    Returns:
        {
            # --- Type 1 (Fabrication / Syntactic)
            'citation_hallucination_rate': float,
            'fabrication_rate': float,  # alias of citation_hallucination_rate

            # --- Type 2 (Misattribution / Non-GT among retrieved)
            'misattribution_rate': float,  # |(C ∩ R) \\ GT| / |C|
            'misattribution_rate_conditional': float,  # |(C ∩ R) \\ GT| / |C ∩ R|
            'non_gt_citation_rate': float,  # alias of misattribution_rate

            # --- Standard evidence metrics
            'citation_precision': float,
            'citation_recall': float,

            # --- Counts (for debugging and analysis)
            'num_citations': int,
            'num_retrieved': int,
            'num_valid_retrieved_citations': int,  # |C ∩ R|
            'num_hallucinated_citations': int,      # |C \\ R|
            'num_misattributed_citations': int,     # |(C ∩ R) \\ GT|
        }
    
    Example:
        retrieved = ["123", "456", "789"]
        cited = ["123", "456", "999"]
        relevant = ["123", "789"]
        
        hallucinated = {"999"}  # cited but NOT retrieved (syntactic violation)
        hallucination_rate = 1/3 = 0.333 (33.3%)
        
        correct_citations = {"123"}  # cited AND relevant
        citation_precision = 1/3 = 0.333 (33.3%)
        citation_recall = 1/2 = 0.5 (50%)
    """
    # Convert to sets for efficient set operations
    cited_set = set(citations)
    retrieved_set = set(retrieved)
    relevant_set = set(relevant_articles)
    
    # PRIMARY METRIC: Citation Hallucination Rate (TYPE 1: FABRICATION / SYNTACTIC CHECK)
    # Definition: Citations that appear in answer but were NOT in retrieved set
    # This is a HARD CONSTRAINT violation - system cited articles it never retrieved
    hallucinated_citations = cited_set - retrieved_set  # Set difference: cited \ retrieved
    hallucination_rate = len(hallucinated_citations) / len(cited_set) if len(cited_set) > 0 else 0.0

    # STRICT GT comparator: citations not present in the gold set at all.
    # This is the apples-to-apples metric when comparing against Layer 2 perfect-context runs,
    # because in e2e retrieval the retrieved context is noisy and differs from GT.
    gt_hallucinated_citations = cited_set - relevant_set
    gt_hallucination_rate = len(gt_hallucinated_citations) / len(cited_set) if len(cited_set) > 0 else 0.0
    
    # NEW METRIC: Misattribution Rate (TYPE 2: SEMANTIC HALLUCINATION)
    # Definition: Citations that are VALIDLY retrieved (syntactically correct) but NOT IN GT.
    # NOTE: In legal tasks, GT can be incomplete (alternative valid statutes may exist). For paper writing,
    # interpret this as "Non-GT citation rate" unless you validate GT completeness.
    #
    # We report two versions:
    # - misattribution_rate: overall, normalized by |C|
    # - misattribution_rate_conditional: conditioned on syntactically-valid citations |C ∩ R|
    validly_retrieved_citations = cited_set & retrieved_set
    misattributed_citations = validly_retrieved_citations - relevant_set
    # Metric: Misattribution Rate Conditional (TYPE 2: SEMANTIC ERROR)
    # Definition: Of the citations that were valid (in retrieved set), how many were NOT in GT?
    # This isolates semantic selection error from syntactic hallucination.
    misattribution_rate_conditional = (
        len(misattributed_citations) / len(validly_retrieved_citations)
        if len(validly_retrieved_citations) > 0 else 0.0
    )
    misattribution_rate = (
        len(misattributed_citations) / len(cited_set)
        if len(cited_set) > 0 else 0.0
    )

    # SECONDARY METRIC: Citation Precision (SEMANTIC CHECK)
    # Definition: What percentage of citations are actually relevant (correct)?
    # This measures answer quality, not constraint violation
    correct_citations = cited_set & relevant_set  # Set intersection: cited ∩ relevant
    citation_precision = len(correct_citations) / len(cited_set) if len(cited_set) > 0 else 0.0
    
    # SECONDARY METRIC: Citation Recall (COVERAGE CHECK)
    # Definition: What percentage of relevant articles were cited?
    # This measures completeness of answer
    citation_recall = len(correct_citations) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return {
        'citation_hallucination_rate': hallucination_rate,  # Core thesis metric (Fabrication/Type 1)
        'fabrication_rate': hallucination_rate,             # Explicit alias for Type 1
        'citation_hallucination_rate_gt': gt_hallucination_rate,  # Strict GT comparator for Layer 2
        'misattribution_rate': misattribution_rate,         # Type 2 overall (normalized by |C|)
        'misattribution_rate_conditional': misattribution_rate_conditional,  # Type 2 conditioned on |C ∩ R|
        'non_gt_citation_rate': misattribution_rate,        # Alias
        
        'citation_precision': citation_precision,
        'citation_recall': citation_recall,
        'num_citations': len(cited_set),
        'num_valid_retrieved_citations': len(validly_retrieved_citations),
        'num_hallucinated_citations': len(hallucinated_citations),
        'num_misattributed_citations': len(misattributed_citations),
        'num_retrieved': len(retrieved_set)
    }


def calculate_retrieval_metrics(retrieved: List[str], relevant_articles: List[str]) -> Dict[str, float]:
    """Calculate retrieval quality against COLIEE GT citations.

    These metrics explain upstream failures (missed GT articles) that cap generation/citation recall.
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant_articles)
    overlap = retrieved_set & relevant_set

    hit = 1.0 if overlap else 0.0
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(relevant_set) if relevant_set else 0.0

    return {
        'retrieval_hit': hit,
        'retrieval_precision': precision,
        'retrieval_recall': recall,
        'num_retrieved': len(retrieved_set),
        'num_gt': len(relevant_set),
        'num_overlap': len(overlap),
    }


def evaluate_rag_system(
    system: RAGSystem,
    queries: List[Dict[str, Any]],
    system_name: str,
    dataset: str = "coliee",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a RAG system on citation constraints and answer quality.
    
    Args:
        system: RAG system instance (implements RAGSystem interface)
        queries: List of query dictionaries
        system_name: Name for logging
        dataset: Dataset name used to gate answer-accuracy scoring (Y/N only for coliee/housing_qa)
    Returns:
        (aggregate_metrics, per_query_results)
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {system_name.upper()} System")
    print(f"{'='*70}")
    
    print(f"\n📊 Evaluating on {len(queries)} queries")
    
    per_query_results = []
    
    # Metrics to aggregate
    hallucination_rates = []
    hallucination_rates_gt = []
    misattribution_rates = []
    misattribution_rates_conditional = []
    citation_precisions = []
    citation_recalls = []
    answer_correctness = []
    answer_coverage = []
    retrieval_hits = []
    retrieval_precisions = []
    retrieval_recalls = []
    retrieval_times = []
    generation_times = []
    total_times = []
    hallucination_num_used = 0
    precision_num_used = 0
    
    for i, query_data in enumerate(queries, 1):
        query_id = query_data['query_id']
        question = query_data['question']
        ground_truth = query_data['ground_truth_answer']
        relevant_articles = query_data['relevant_articles']
        
        print(f"\n[{i}/{len(queries)}] Processing {query_id}...")
        print(f"  Question: {question[:100]}...")
        print(f"  GT Citations: {len(relevant_articles)} articles")
        
        # Query the system
        t_start = time.time()
        result = system.query(question)
        total_time = time.time() - t_start
        
        # Extract results
        answer = result.get('answer', '')
        citations = result.get('citations', [])
        retrieved = result.get('retrieved', [])
        json_parse_success = int(result.get('json_parse_success', 1))
        retrieval_time = result.get('retrieval_time', 0.0)
        generation_time = result.get('generation_time', 0.0)

        # Answer accuracy is only calculated for datasets with Y/N answers
        predicted_label = None
        gt_label = None
        if dataset in ["coliee", "housing_qa"]:
            predicted_label = normalize_yn_answer(answer) if json_parse_success == 1 else None
            has_gt_answer = ground_truth is not None and str(ground_truth).strip() != ''
            gt_label = normalize_yn_answer(str(ground_truth)) if has_gt_answer else None

            if gt_label is not None:
                is_correct = 1 if predicted_label == gt_label else 0
            else:
                is_correct = None
            is_parsed = 1 if predicted_label is not None else 0
        else:
            is_correct = None
            is_parsed = json_parse_success
        
        print(f"  Retrieved: {len(retrieved)} articles")
        print(f"  Citations: {len(citations)} articles")
        
        # Calculate metrics
        metrics = calculate_citation_metrics(citations, retrieved, relevant_articles)
        retrieval_metrics = calculate_retrieval_metrics(retrieved, relevant_articles)
        
        print(f"  Hallucination (vs retrieved, Type 1): {metrics['citation_hallucination_rate']*100:.1f}%")
        print(f"  Hallucination (vs GT): {metrics['citation_hallucination_rate_gt']*100:.1f}%")
        print(f"  Misattribution (Type 2): {metrics['misattribution_rate']*100:.1f}%")
        print(f"  Misattribution|Valid (Type 2): {metrics['misattribution_rate_conditional']*100:.1f}%")
        print(f"  Citation Precision: {metrics['citation_precision']*100:.1f}%")
        print(f"  Citation Recall: {metrics['citation_recall']*100:.1f}%")
        print(f"  Retrieval Hit@K: {retrieval_metrics['retrieval_hit']*100:.1f}%")
        print(f"  Retrieval Recall@K: {retrieval_metrics['retrieval_recall']*100:.1f}%")
        gt_print = gt_label if gt_label is not None else 'N/A'
        print(f"  Answer (GT→Pred): {gt_print} → {predicted_label if predicted_label is not None else 'UNK'}")
        # Debug-friendly inspection: show GT vs LLM citation IDs directly.
        gt_list = sorted(set(relevant_articles))
        llm_list = sorted(set(citations))
        print(f"  GT: {gt_list}")
        print(f"  LLM: {llm_list}")
        print(f"  Time: {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        # Aggregate logic
        # - Hallucination is conditional on structured parse success.
        # - Precision/Recall are query-level and averaged across all queries.
        if json_parse_success == 1:
            hallucination_rates.append(metrics['citation_hallucination_rate'])
            hallucination_rates_gt.append(metrics['citation_hallucination_rate_gt'])
            hallucination_num_used += 1
            misattribution_rates.append(metrics['misattribution_rate'])

        citation_precisions.append(metrics['citation_precision'])
        precision_num_used += 1

        if metrics['num_valid_retrieved_citations'] > 0:
            misattribution_rates_conditional.append(metrics['misattribution_rate_conditional'])

        # Recall is calculated globally across all queries (0 citations = 0% recall, correctly penalizing)
        citation_recalls.append(metrics['citation_recall'])
        
        retrieval_hits.append(retrieval_metrics['retrieval_hit'])
        retrieval_precisions.append(retrieval_metrics['retrieval_precision'])
        retrieval_recalls.append(retrieval_metrics['retrieval_recall'])
        
        if is_correct is not None:
            answer_correctness.append(is_correct)
            answer_coverage.append(is_parsed)
        
        retrieval_times.append(retrieval_time)
        generation_times.append(generation_time)
        total_times.append(total_time)
        
        # Store per-query result
        per_query_results.append({
            'query_id': query_id,
            'question': question,
            'ground_truth_answer': ground_truth,
            'system_answer': answer,
            'predicted_answer': predicted_label,
            'answer_correct': is_correct,
            'json_parse_success': json_parse_success,
            'gt_citations': relevant_articles,
            'retrieved_articles': retrieved,
            'cited_articles': citations,
            'metrics': metrics,
            'retrieval_metrics': retrieval_metrics,
            'timing': {
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': total_time
            }
        })
    
    # Compute aggregate statistics
    aggregate_metrics = {
        'citation_hallucination_rate': {  # Type 1: Fabrication
            'mean': mean(hallucination_rates) if hallucination_rates else 0.0,
            'std': stdev(hallucination_rates) if len(hallucination_rates) > 1 else 0.0,
            'min': min(hallucination_rates) if hallucination_rates else 0.0,
            'max': max(hallucination_rates) if hallucination_rates else 0.0,
            'denominator_queries': hallucination_num_used,
        },
        'citation_hallucination_rate_gt': {  # Strict GT comparator for Layer 2
            'mean': mean(hallucination_rates_gt) if hallucination_rates_gt else 0.0,
            'std': stdev(hallucination_rates_gt) if len(hallucination_rates_gt) > 1 else 0.0,
            'min': min(hallucination_rates_gt) if hallucination_rates_gt else 0.0,
            'max': max(hallucination_rates_gt) if hallucination_rates_gt else 0.0,
            'denominator_queries': hallucination_num_used,
        },
        'misattribution_rate': {  # Type 2: Semantic Hallucination
            'mean': mean(misattribution_rates) if misattribution_rates else 0.0,
            'std': stdev(misattribution_rates) if len(misattribution_rates) > 1 else 0.0,
            'min': min(misattribution_rates) if misattribution_rates else 0.0,
            'max': max(misattribution_rates) if misattribution_rates else 0.0
        },
        'misattribution_rate_conditional': {
            'mean': mean(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0,
            'std': stdev(misattribution_rates_conditional) if len(misattribution_rates_conditional) > 1 else 0.0,
            'min': min(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0,
            'max': max(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0
        },
        'citation_precision': {
            'mean': mean(citation_precisions) if citation_precisions else 0.0,
            'std': stdev(citation_precisions) if len(citation_precisions) > 1 else 0.0,
            'min': min(citation_precisions) if citation_precisions else 0.0,
            'max': max(citation_precisions) if citation_precisions else 0.0,
            'denominator_queries': precision_num_used,
        },
        'citation_recall': {
            'mean': mean(citation_recalls) if citation_recalls else 0.0,
            'std': stdev(citation_recalls) if len(citation_recalls) > 1 else 0.0,
            'min': min(citation_recalls) if citation_recalls else 0.0,
            'max': max(citation_recalls) if citation_recalls else 0.0,
            'denominator_queries': len(citation_recalls),
        },
        'answer_accuracy': {
            'mean': mean(answer_correctness) if answer_correctness else 0.0,
            'std': stdev(answer_correctness) if len(answer_correctness) > 1 else 0.0,
            'num_scored': len(answer_correctness),
            'num_total': len(queries)
        },
        'json_compliance': {
            'mean': mean([r['json_parse_success'] for r in per_query_results]) if per_query_results else 0.0,
            'count_success': sum(r['json_parse_success'] for r in per_query_results),
            'count_fail': len(per_query_results) - sum(r['json_parse_success'] for r in per_query_results),
            'total': len(per_query_results),
        },
        'answer_coverage': {
            # Fraction of queries where we could confidently parse a Y/N from the model output.
            'mean': mean(answer_coverage) if answer_coverage else 0.0,
            'num_scored': len(answer_correctness),
            'num_total': len(queries)
        },
        'retrieval_hit': {
            'mean': mean(retrieval_hits) if retrieval_hits else 0.0,
            'std': stdev(retrieval_hits) if len(retrieval_hits) > 1 else 0.0,
        },
        'retrieval_precision': {
            'mean': mean(retrieval_precisions) if retrieval_precisions else 0.0,
            'std': stdev(retrieval_precisions) if len(retrieval_precisions) > 1 else 0.0,
        },
        'retrieval_recall': {
            'mean': mean(retrieval_recalls) if retrieval_recalls else 0.0,
            'std': stdev(retrieval_recalls) if len(retrieval_recalls) > 1 else 0.0,
        },
        'retrieval_time': {
            'mean': mean(retrieval_times),
            'std': stdev(retrieval_times) if len(retrieval_times) > 1 else 0.0,
            'total': sum(retrieval_times)
        },
        'generation_time': {
            'mean': mean(generation_times),
            'std': stdev(generation_times) if len(generation_times) > 1 else 0.0,
            'total': sum(generation_times)
        },
        'total_time': {
            'mean': mean(total_times),
            'std': stdev(total_times) if len(total_times) > 1 else 0.0,
            'total': sum(total_times)
        }
    }
    
    print(f"\n{'='*70}")
    print(f"📊 AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Answer Accuracy: {aggregate_metrics['answer_accuracy']['mean']*100:.2f}% (scored {aggregate_metrics['answer_accuracy']['num_scored']}/{aggregate_metrics['answer_accuracy']['num_total']})")
    print(f"Citation Hallucination Rate (vs retrieved, Type 1): {aggregate_metrics['citation_hallucination_rate']['mean']*100:.2f}%")
    print(f"Citation Hallucination Rate (vs GT): {aggregate_metrics['citation_hallucination_rate_gt']['mean']*100:.2f}%")
    print(f"Misattribution Rate (Type 2): {aggregate_metrics['misattribution_rate']['mean']*100:.2f}%")
    print(f"Misattr.|Valid (Type 2): {aggregate_metrics['misattribution_rate_conditional']['mean']*100:.2f}%")
    print(f"Citation Precision: {aggregate_metrics['citation_precision']['mean']*100:.2f}%")
    print(f"Citation Recall: {aggregate_metrics['citation_recall']['mean']*100:.2f}%")
    print(f"Retrieval Hit@K: {aggregate_metrics['retrieval_hit']['mean']*100:.2f}% ± {aggregate_metrics['retrieval_hit']['std']*100:.2f}%")
    print(f"Retrieval Recall@K: {aggregate_metrics['retrieval_recall']['mean']*100:.2f}% ± {aggregate_metrics['retrieval_recall']['std']*100:.2f}%")
    print(f"Average Total Time: {aggregate_metrics['total_time']['mean']:.2f}s ± {aggregate_metrics['total_time']['std']:.2f}s")
    print(f"{'='*70}\n")
    
    return aggregate_metrics, per_query_results


def save_results(
    output_path: Path,
    system_name: str,
    aggregate_metrics: Dict[str, Any],
    per_query_results: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """Save evaluation results to JSON file"""
    results = {
        'metadata': {
            'system_name': system_name,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(per_query_results),
            **metadata
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': per_query_results
    }
    
    # Delete old result file if it exists
    if output_path.exists():
        output_path.unlink()
        print(f"🗑️  Deleted old result: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Template 2: End-to-End RAG Evaluation")

    dataset_choices = available_datasets()
    
    # Data paths
    parser.add_argument('--dataset', type=str, default='coliee', choices=dataset_choices,
                       help='Dataset adapter to use for corpus/query loading')
    parser.add_argument('--corpus', type=str, default=None,
                       help='Path to corpus XML file')
    parser.add_argument('--queries', type=str, default=None,
                       help='Path to queries XML file')
    parser.add_argument('--year', type=str, default='R06',
                       help='COLIEE year identifier used for default query file selection when --dataset coliee and --queries is not provided')
    
    # System configuration
    system_choices = available_systems()
    parser.add_argument('--system', type=str, default='aurag',
                       choices=system_choices,
                       help='RAG system to evaluate')
    parser.add_argument('--embedding', type=str, default='multilingual',
                       help='Embedding model identifier')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for embedding model (cpu | cuda | mps). Use cuda on HPC GPU nodes.')
    parser.add_argument('--llm-model', type=str,
                       default='models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
                       help='Path to LLM model (GGUF format) - Llama-3.1-8B for legal context')
    parser.add_argument('--n-ctx', type=int, default=DEFAULT_N_CTX,
                       help='Model context window used by llama.cpp')
    parser.add_argument('--max-tokens', type=str, default='auto',
                       help="Generation token cap. Use integer or 'auto' to use model max available tokens per prompt.")
    
    # Evaluation parameters
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--retrieval-mode', type=str, default='dense', choices=['dense', 'hybrid'],
                       help='Layer-1 retrieval strategy for end-to-end runs')
    parser.add_argument('--bm25-weight', type=float, default=0.5,
                       help='BM25 weight in weighted-RRF when retrieval-mode=hybrid')
    parser.add_argument('--rrf-k', type=int, default=60,
                       help='RRF constant (larger = flatter rank decay)')
    parser.add_argument('--child-chunk-size', type=int, default=300,
                       help='Chunk token size for flat/hierarchical chunking (default 300; try 512 for flat_prompt baseline)')
    parser.add_argument('--child-chunk-overlap', type=int, default=90,
                       help='Chunk token overlap (default 90)')
    parser.add_argument('--citation-mode', type=str, default='short', choices=['short', 'both'],
                       help='Citation reference format for prompt-only baseline (short=article IDs, both=full+short)')
    parser.add_argument('--sample-queries', type=int, default=None,
                       help='Limit to first N queries (for testing)')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Delete and rebuild the persisted vector DB before running (recommended for clean paper runs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save results')
    
    args = parser.parse_args()

    max_tokens_text = str(args.max_tokens).strip().lower()
    if max_tokens_text in ('auto', 'max', 'model_max'):
        parsed_max_tokens = None
    else:
        parsed_max_tokens = int(max_tokens_text)
        if parsed_max_tokens <= 0:
            parsed_max_tokens = None

    # Set seed for reproducibility
    random.seed(args.seed)
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # For COLIEE, map --year to default query path unless user explicitly overrides --queries.
    provided_queries = args.queries
    if args.dataset == 'coliee' and not args.queries:
        year_upper = str(args.year).strip().upper()
        year_query_candidate = f"benchmark/COLIEE/simple/simple_{year_upper}_jp.xml"
        year_query_candidate_path = workspace_root / year_query_candidate
        if year_query_candidate_path.exists():
            provided_queries = year_query_candidate
        else:
            print(
                f"⚠️  COLIEE year file not found for {year_upper}: {year_query_candidate}. "
                "Falling back to dataset default queries."
            )

    # Resolve dataset-specific paths via standardized dataset defaults/overrides.
    corpus_path, queries_path = resolve_dataset_paths(
        dataset=args.dataset,
        workspace_root=workspace_root,
        provided_corpus=args.corpus,
        provided_queries=provided_queries,
    )
    
    corpus, queries, dataset_metadata = load_dataset(
        dataset=args.dataset,
        workspace_root=workspace_root,
        corpus_path=corpus_path,
        queries_path=queries_path,
    )

    if args.sample_queries and args.sample_queries > 0:
        queries = random.sample(queries, min(args.sample_queries, len(queries)))
    
    # Embedding model mapping
    EMBEDDINGS_CONFIG = {
        'multilingual': 'intfloat/multilingual-e5-small',  # same encoder as Layer-1 retrieval (evaluate_retrieval.py)
        'jp-bge': 'BAAI/bge-base-ja-v1.5',
        'en-bge': 'BAAI/bge-small-en-v1.5'
    }
    embedding_model = EMBEDDINGS_CONFIG.get(args.embedding, args.embedding)
    
    # Setup system
    print(f"\n{'='*70}")
    print(f"TEMPLATE 2: END-TO-END RAG EVALUATION")
    print(f"{'='*70}")
    print(f"\nSystem: {args.system.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Corpus: {len(corpus)} articles")
    print(f"Queries: {len(queries)} questions")
    print(f"Embedding: {embedding_model}")
    print(f"LLM: {args.llm_model}")
    print(f"Top-K: {args.top_k}")
    print(f"Chunk size / overlap: {args.child_chunk_size} / {args.child_chunk_overlap}")
    print(f"Model Context (n_ctx): {args.n_ctx}")
    print(f"Generation Max Tokens: {'auto' if parsed_max_tokens is None else parsed_max_tokens}")
    print(f"Citation mode: {args.citation_mode}")
    print(f"Retrieval Mode: {args.retrieval_mode}")
    if args.retrieval_mode == 'hybrid':
        print(f"BM25 Weight: {args.bm25_weight}")
        print(f"RRF-k: {args.rrf_k}")
    if args.sample_queries:
        print(f"Sample Size: {args.sample_queries} queries")
    
    # Initialize system
    system = create_system(args.system)
    dataset_suffix = str(args.year).strip().upper() if args.dataset == 'coliee' else 'default'
    db_path = output_dir / f"vector_db_{args.system}_{args.dataset}_{dataset_suffix}_{args.embedding.replace('/', '_')}"

    system.setup(
        corpus=corpus,
        embedding_model=embedding_model,
        llm_model_path=args.llm_model,
        db_path=db_path,
        top_k=args.top_k,
        rebuild_index=args.rebuild_index,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        context_budget=12000,
        retrieval_mode=args.retrieval_mode,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
        seed=args.seed,
        n_ctx=args.n_ctx,
        max_tokens=parsed_max_tokens,
        n_gpu_layers=-1,
        citation_mode=args.citation_mode,
        device=args.device,
    )
    
    # Run evaluation
    aggregate_metrics, per_query_results = evaluate_rag_system(
        system=system,
        queries=queries,
        system_name=args.system,
        dataset=args.dataset,
    )
    
    # Save results
    output_filename = f"evaluation_results_end2end_{args.system}_k{args.top_k}.json"
    if args.sample_queries:
        output_filename = output_filename.replace('.json', f'_sample{args.sample_queries}.json')
    
    output_path = output_dir / output_filename
    
    save_results(
        output_path=output_path,
        system_name=args.system,
        aggregate_metrics=aggregate_metrics,
        per_query_results=per_query_results,
        metadata={
            **system.get_metadata(),
            **dataset_metadata,
            'dataset_name': args.dataset,
            'coliee_year': str(args.year).strip().upper() if args.dataset == 'coliee' else None,
            'queries_path_resolved': str(queries_path),
            'embedding_model': embedding_model,
            'child_chunk_size': args.child_chunk_size,
            'child_chunk_overlap': args.child_chunk_overlap,
            'top_k': args.top_k,
            'n_ctx': args.n_ctx,
            'max_tokens': args.max_tokens,
            'citation_mode': args.citation_mode,
            'seed': args.seed,
        }
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"📄 Results: {output_path}")


if __name__ == "__main__":
    main()
