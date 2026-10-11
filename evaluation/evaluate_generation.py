#!/usr/bin/env python3
"""
Template 2b: Generation Evaluation with Ideal Retrieval (GT)
=========================================================

Evaluate generation quality and citation behavior when the LLM is given
an ideal retrieval list (ground-truth articles). This isolates Layer 2.

Compares:
- prompt: Zero-shot-CoT + citation + JSON (no grammar constraints)
- rdg: RDG (grammar-constrained JSON) with the same prompt

Usage:
    Minimal CLI (quick run):
    python3 evaluation/evaluate_generation.py --mode rdg --dataset coliee

    Complete CLI (reproducible and model-agnostic):
    python3 evaluation/evaluate_generation.py \
        --dataset coliee \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --mode rdg \
        --citation-mode short \
        --llm-model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
        --n-ctx 16384 \
        --max-tokens auto \
        --n-gpu-layers -1 \
        --seed 42

    Compare RDG vs prompt baseline with the same budget:
    python3 evaluation/evaluate_generation.py --mode rdg --llm-model <model.gguf> --n-ctx <ctx> --max-tokens auto
    python3 evaluation/evaluate_generation.py --mode prompt --llm-model <model.gguf> --n-ctx <ctx> --max-tokens auto

    Smoke test (10 queries):
    python3 evaluation/evaluate_generation.py --mode rdg --sample-queries 10 --max-tokens auto
"""

import argparse
import json
import random
import time
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Dict, List, Any, Tuple, Optional, Literal
from statistics import mean, stdev

try:
    import openai
except ImportError:
    openai = None

from langchain_core.documents import Document

# Add project root and src/ to path for local imports
import sys
EVAL_PATH = Path(__file__).parent
ROOT_PATH = EVAL_PATH.parent
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(ROOT_PATH))
sys.path.insert(0, str(SRC_PATH))

# Local imports
from evaluation.evaluate_end2end import (
    normalize_yn_answer,
    calculate_citation_metrics,
)
from evaluation.adapters import available_datasets, load_dataset
from evaluation.dataset_defaults import resolve_dataset_paths

from RDG import (
    get_rdg_pipeline,
    DEFAULT_N_CTX,
    Citation,
    StructuredReasoning,
    _parse_json_output,
    collect_validated_citations,
    ReferenceExtractor,
    RDGPipeline,
)


def build_documents_from_gt(corpus: Dict[str, Dict[str, str]], gt_articles: List[str]) -> List[Document]:
    docs = []
    for article_id in gt_articles:
        item = corpus.get(article_id)
        if not item:
            continue
        docs.append(Document(
            page_content=item['text'],
            metadata={
                'contract_id': article_id,
                'section_num': article_id,
                'section_title': 'Article',
            }
        ))
    return docs


def extract_contract_id(ref: str) -> str:
    """
    Robustly extract the contract_id (Article Number) from various citation formats.
    Handles:
    - Full Ref: "123_Section_1_Title" -> "123"
    - Short Ref: "123_sec_1" -> "123"
    - Raw ID: "123" -> "123"
    """
    if "_Section_" in ref:
        return ref.split("_Section_")[0]
    if "_sec_" in ref:
        return ref.split("_sec_")[0]
    return ref.strip()


def extract_article_ids_from_citations(citations: List[Citation]) -> List[str]:
    ids = []
    for c in citations:
        # Use the robust extraction on the raw reference_id or contract_id
        if c.contract_id and c.contract_id != "Unknown":
             ids.append(extract_contract_id(c.contract_id))
        elif c.reference_id:
             ids.append(extract_contract_id(c.reference_id))
    return sorted(set(ids))


def gpt4o_generate(
    rdg,
    question: str,
    documents: List[Document],
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    citation_mode: Literal["short", "both"] = "short",
):
    if not openai:
        raise ImportError("The 'openai' package is required for GPT-4o evaluation. Please run `pip install openai`.")
        
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    
    # FAIRNESS: Use the exact same context extractor as RDG
    valid_refs_full = rdg.extractor.extract_references(documents)

    context_parts = []
    for doc in documents:
        # Reconstruct Full Ref
        c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
        s_num = doc.metadata.get('section_num', 'N/A')
        s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
        full_ref = f"{c_id}_Section_{s_num}_{s_title}"
        context_parts.append(f"[{full_ref}]\n{doc.page_content.strip()}")

    context_str = "\n\n---\n\n".join(context_parts)

    # FAIRNESS: Use the exact same prompt generation logic as RDG
    prompt_str = rdg._build_prompt(question, valid_refs_full, context_str, answer_mode="yn")
    
    # Parse Llama-3 format into OpenAI messages format
    system_part = prompt_str.split("<|start_header_id|>system<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
    user_part = prompt_str.split("<|start_header_id|>user<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part}
        ],
        temperature=temperature,
        response_format={ "type": "json_object" },
    )
    
    raw_text = response.choices[0].message.content.strip()
    output_dict = _parse_json_output(raw_text)
    
    if not output_dict:
        return {
            'answer': raw_text,
            'reasoning': StructuredReasoning([]),
            'citations': [],
            'json_parse_success': 0,
            'raw_output_text': raw_text,
            'effective_max_tokens': None,
        }
    reasoning = StructuredReasoning.from_dict(output_dict.get('reasoning', {}))
    validated_citations = collect_validated_citations(output_dict, valid_refs_full, citation_mode=citation_mode)

    return {
        'answer': str(output_dict.get('answer', '')),
        'reasoning': reasoning,
        'citations': validated_citations,
        'json_parse_success': 1,
        'raw_output_text': raw_text,
        'effective_max_tokens': None,
    }


def prompt_only_generate(
    rdg,
    question: str,
    documents: List[Document],
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    citation_mode: Literal["short", "both"] = "short",
):
    # FAIRNESS: Use the exact same context construction as RDG (Full Headers)
    # This ensures both models see the same information (including Section Titles).
    valid_refs_full = rdg.extractor.extract_references(documents)

    context_parts = []
    for doc in documents:
        # Reconstruct Full Ref (same as RDG.py)
        c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
        s_num = doc.metadata.get('section_num', 'N/A')
        s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
        full_ref = f"{c_id}_Section_{s_num}_{s_title}"
        context_parts.append(f"[{full_ref}]\n{doc.page_content.strip()}")

    # Build prompt using Full Refs
    # (The prompt text asks for Short IDs, but we will accept Full IDs in post-processing)
    prompt = rdg._build_prompt(
        question,
        valid_refs_full,
        "\n\n---\n\n".join(context_parts),
        answer_mode="yn",
    )
    effective_max_tokens = rdg.resolve_max_tokens(prompt, max_tokens=max_tokens)

    output = rdg.llm(
        prompt,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        repeat_penalty=1.12,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

    raw_text = output['choices'][0]['text'].strip()
    output_dict = _parse_json_output(raw_text)
    
    if not output_dict:
        # Strict mode: broken JSON is a hard failure.
        return {
            'answer': raw_text,
            'reasoning': StructuredReasoning([]),
            'citations': [],
            'json_parse_success': 0,
            'raw_output_text': raw_text,
            'effective_max_tokens': effective_max_tokens,
        }
    reasoning = StructuredReasoning.from_dict(output_dict.get('reasoning', {}))
    validated_citations = collect_validated_citations(output_dict, valid_refs_full, citation_mode=citation_mode)

    return {
        'answer': output_dict.get('answer', ''),
        'reasoning': reasoning,
        'citations': validated_citations,
        'json_parse_success': 1,
        'raw_output_text': raw_text,
        'effective_max_tokens': effective_max_tokens,
    }


def parse_max_tokens_arg(value: str) -> Optional[int]:
    text = str(value).strip().lower()
    if text in ('auto', 'max', 'model_max'):
        return None
    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def evaluate_generation(
    mode: str,
    corpus: Dict[str, Dict[str, str]],
    queries: List[Dict[str, Any]],
    llm_model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    max_tokens: Optional[int],
    citation_mode: Literal["short", "both"] = "short",
    seed: int = 42,
    sample_size: int = None,
    dataset: str = "coliee",
):
    if sample_size:
        queries = queries[:sample_size]
        print(f"\n📊 Testing on {len(queries)} sample queries")
    else:
        print(f"\n📊 Evaluating on {len(queries)} queries")

    # Validate Layer 2 setup: retrieved should equal relevant_articles (GT context)
    print(f"✓ Layer 2: Perfect retrieval (GT context isolation)")

    # Always initialize RDG context even for OpenAI to ensure 100% uniformity 
    # of the prompting logic and reference extraction phase.
    rdg = get_rdg_pipeline(model_path=llm_model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=seed)

    per_query_results = []
    hallucination_rates = []
    citation_precisions = []
    citation_recalls = []
    answer_correctness = []
    generation_times = []
    hallucination_num_used = 0
    precision_num_used = 0
    raw_outputs = []
    skipped_queries = []  # Track queries with missing corpus documents

    for i, query_data in enumerate(queries, 1):
        query_id = query_data['query_id']
        question = query_data['question']
        ground_truth = query_data['ground_truth_answer']
        relevant_articles = query_data['relevant_articles']

        print(f"\n[{i}/{len(queries)}] {query_id}")

        # Build documents from GT (Layer 2: Perfect retrieval)
        retrieved = sorted(set(relevant_articles))
        assert len(retrieved) > 0, f"❌ Layer 2 BROKEN: Query {query_id} has no GT articles! Parser fix may have failed."
        docs = build_documents_from_gt(corpus, retrieved)
        
        # Guard: Skip queries with missing corpus documents (e.g., CUAD Unicode mismatch)
        if len(retrieved) > 0 and len(docs) == 0:
            print(f"  ⚠️  SKIPPED: Query references {len(retrieved)} GT article(s) but none found in corpus.")
            skipped_queries.append({
                'query_id': query_id,
                'reason': 'missing_corpus_documents',
                'referenced_articles': retrieved,
            })
            continue

        t_start = time.time()
        if mode == "rdg":
            structured = rdg.generate(question=question, documents=docs, max_tokens=max_tokens, answer_mode="yn")
            answer = structured.answer
            citations = extract_article_ids_from_citations(structured.citations)
            json_parse_success = 1
            if structured.metadata and 'json_parse_success' in structured.metadata:
                json_parse_success = int(structured.metadata['json_parse_success'])
            raw_output_text = answer
            if structured.metadata and 'raw_output_text' in structured.metadata:
                raw_output_text = str(structured.metadata['raw_output_text'])
            effective_max_tokens = None
            if structured.metadata and 'effective_max_tokens' in structured.metadata:
                effective_max_tokens = int(structured.metadata['effective_max_tokens'])
        elif mode == "prompt":
            baseline = prompt_only_generate(
                rdg,
                question,
                docs,
                max_tokens=max_tokens,
                temperature=0.0,
                citation_mode=citation_mode,
            )
            answer = baseline['answer']
            citations = extract_article_ids_from_citations(baseline['citations'])
            json_parse_success = int(baseline.get('json_parse_success', 0))
            raw_output_text = str(baseline.get('raw_output_text', answer))
            effective_max_tokens = baseline.get('effective_max_tokens')
        elif mode == "gpt4o":
            baseline = gpt4o_generate(
                rdg,
                question,
                docs,
                model_name="gpt-4o",
                temperature=0.0,
                citation_mode=citation_mode,
            )
            answer = baseline['answer']
            citations = extract_article_ids_from_citations(baseline['citations'])
            json_parse_success = int(baseline.get('json_parse_success', 0))
            raw_output_text = str(baseline.get('raw_output_text', answer))
            effective_max_tokens = baseline.get('effective_max_tokens')

        generation_time = time.time() - t_start

        # Answer evaluation: only for datasets with Y/N binary answers (COLIEE, housing_qa)
        # CUAD doesn't have binary answer labels, so skip answer-accuracy scoring for CUAD.
        predicted_label = normalize_yn_answer(answer) if json_parse_success == 1 else None
        is_correct = None
        gt_label = None
        
        if dataset in ["coliee", "housing_qa"]:  # Datasets with binary Y/N answers
            has_gt_answer = ground_truth is not None and str(ground_truth).strip() != ''
            if has_gt_answer:
                gt_label = normalize_yn_answer(str(ground_truth))
                if gt_label is not None:
                    is_correct = 1 if predicted_label == gt_label else 0
        # For CUAD, is_correct remains None (not scored)

        metrics = calculate_citation_metrics(citations, retrieved, relevant_articles)
        
        gt_print = gt_label if gt_label is not None else 'N/A'
        status = '✓' if is_correct == 1 else ('✗' if is_correct == 0 else '-')
        print(f"  Answer: {gt_print} → {predicted_label or '?'} {status}")
        print(
            f"  Hallucination: {metrics['citation_hallucination_rate']*100:.1f}% | "
            f"Precision: {metrics['citation_precision']*100:.1f}% | "
            f"Recall: {metrics['citation_recall']*100:.1f}%"
        )
        print(f"  Time: {generation_time:.2f}s")

        # Query-level metrics: precision/recall are averaged over all queries.
        # Hallucination is conditioned on successful JSON parsing.
        if json_parse_success == 1:
            hallucination_rates.append(metrics['citation_hallucination_rate'])
            hallucination_num_used += 1

        citation_precisions.append(metrics['citation_precision'])
        precision_num_used += 1
        citation_recalls.append(metrics['citation_recall'])
        if is_correct is not None:
            answer_correctness.append(is_correct)
        generation_times.append(generation_time)

        # Store result (minimal necessary fields)
        result_entry = {
            'query_id': query_id,
            'ground_truth_answer': ground_truth,
            'predicted_answer': predicted_label,
            'answer_correct': is_correct,
            'json_parse_success': json_parse_success,
            'gt_citations': relevant_articles,
            'llm_citations': citations,
            'hallucination_rate': metrics['citation_hallucination_rate'],
            'citation_precision': metrics['citation_precision'],
            'citation_recall': metrics['citation_recall'],
            'generation_time': generation_time
        }
        
        # Include raw answer only if normalization changed it
        if answer != predicted_label:
            result_entry['raw_answer'] = answer

        per_query_results.append(result_entry)
        raw_outputs.append({
            'query_id': query_id,
            'mode': mode,
            'json_parse_success': json_parse_success,
            'ground_truth_answer': ground_truth,
            'predicted_answer': predicted_label,
            'raw_output_text': raw_output_text,
            'effective_max_tokens': effective_max_tokens,
        })

    aggregate_metrics = {
        'citation_hallucination_rate': {
            'mean': mean(hallucination_rates) if hallucination_rates else 0.0,
            'std': stdev(hallucination_rates) if len(hallucination_rates) > 1 else 0.0,
            'denominator_queries': hallucination_num_used,
        },
        'skipped_queries': {
            'count': len(skipped_queries),
            'queries': skipped_queries,
        },
        'citation_precision': {
            'mean': mean(citation_precisions) if citation_precisions else 0.0,
            'std': stdev(citation_precisions) if len(citation_precisions) > 1 else 0.0,
            'denominator_queries': precision_num_used,
        },
        'citation_recall': {
            'mean': mean(citation_recalls) if citation_recalls else 0.0,
            'std': stdev(citation_recalls) if len(citation_recalls) > 1 else 0.0,
            'denominator_queries': len(citation_recalls),
        },
        'answer_accuracy': {
            'mean': mean(answer_correctness) if answer_correctness else 0.0,
            'count_correct': sum(answer_correctness),
            'count_wrong': len(answer_correctness) - sum(answer_correctness),
            'total': len(answer_correctness),
        },
        'json_compliance': {
            'mean': mean([r['json_parse_success'] for r in per_query_results]) if per_query_results else 0.0,
            'count_success': sum(r['json_parse_success'] for r in per_query_results),
            'count_fail': len(per_query_results) - sum(r['json_parse_success'] for r in per_query_results),
            'total': len(per_query_results),
        },
        'generation_time': {
            'mean': mean(generation_times) if generation_times else 0.0,
            'std': stdev(generation_times) if len(generation_times) > 1 else 0.0,
            'total': sum(generation_times)
        }
    }

    return aggregate_metrics, per_query_results, raw_outputs


def main():
    parser = argparse.ArgumentParser(description="Generation Evaluation with Ideal Retrieval (GT)")
    dataset_choices = available_datasets()
    parser.add_argument('--dataset', type=str, default='coliee', choices=dataset_choices,
                        help='Dataset adapter to use for corpus/query loading')
    parser.add_argument('--corpus', type=str, default=None, help='Optional corpus path override')
    parser.add_argument('--queries', type=str, default=None, help='Optional queries path override')
    parser.add_argument('--mode', type=str, default='rdg', choices=['rdg', 'prompt', 'gpt4o'])
    parser.add_argument('--llm-model', type=str, default='models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    parser.add_argument('--n-ctx', type=int, default=DEFAULT_N_CTX)
    parser.add_argument('--n-gpu-layers', type=int, default=-1)
    parser.add_argument('--max-tokens', type=str, default='auto',
                        help="Generation token cap. Use an integer, or 'auto' to use each model's max available tokens per prompt.")
    parser.add_argument('--citation-mode', type=str, default='short', choices=['short', 'both'],
                        help="Citation parser mode. Applies only to --mode prompt: 'short' accepts doc_x_sec_y only; 'both' also accepts full-form refs. Ignored in --mode rdg.")
    parser.add_argument('--sample-queries', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

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

    if args.sample_queries > 0:
        queries = random.sample(queries, min(args.sample_queries, len(queries)))

    max_tokens = parse_max_tokens_arg(args.max_tokens)
    citation_mode_applied = 'rdg_internal' if args.mode == 'rdg' else args.citation_mode
    if args.mode == 'rdg' and args.citation_mode != 'short':
        print("ℹ️  --citation-mode is ignored in RDG mode (using RDG internal citation extraction).")

    aggregate_metrics, per_query_results, raw_outputs = evaluate_generation(
        mode=args.mode,
        corpus=corpus,
        queries=queries,
        llm_model_path=args.llm_model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        max_tokens=max_tokens,
        citation_mode=args.citation_mode,
        seed=args.seed,
        sample_size=None,
        dataset=args.dataset
    )

    output = {
        'metadata': {
            'dataset': args.dataset,
            **dataset_metadata,
            'mode': args.mode,
            'citation_mode': citation_mode_applied,
            'citation_mode_requested': args.citation_mode,
            'llm_model': args.llm_model,
            'num_queries': len(per_query_results),
            'max_tokens_requested': args.max_tokens,
            'max_tokens_mode': 'auto' if max_tokens is None else 'fixed',
            'n_ctx': args.n_ctx,
            'seed': args.seed,
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': per_query_results
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"evaluation_results_layer2_generation_{args.mode}_gt.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    raw_output_path = output_path.with_name(f"{output_path.stem}_raw_outputs.json")
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'metadata': output['metadata'],
                'num_queries': len(raw_outputs),
                'raw_outputs': raw_outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ Layer-2 evaluation complete: {output_path}")
    print(f"📝 Raw LLM outputs saved: {raw_output_path}")
    
    print("\n--- Aggregate Metrics ---")
    print(f"Answer Accuracy: {aggregate_metrics['answer_accuracy']['mean']*100:.2f}%")
    print(f"Citation Hallucination: {aggregate_metrics['citation_hallucination_rate']['mean']*100:.2f}%")
    print(f"Citation Precision: {aggregate_metrics['citation_precision']['mean']*100:.2f}%")
    print(f"Citation Recall: {aggregate_metrics['citation_recall']['mean']*100:.2f}%")
    print(f"JSON Compliance: {aggregate_metrics['json_compliance']['mean']*100:.2f}%")


if __name__ == '__main__':
    main()
