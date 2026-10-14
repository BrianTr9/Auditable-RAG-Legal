#!/usr/bin/env python3
"""
Re-score Layer-2 prompt-only outputs under another citation-ID policy, without regenerating.

The citation mode ('short' vs 'both') only changes how attributed IDs are scored, not what the
model generates. So the Lenient row of Table 2 is the 'short' run's raw outputs scored with
citation_mode='both'. Every step reuses the functions of the original run:
GT documents -> ReferenceExtractor -> _parse_json_output -> collect_validated_citations ->
calculate_citation_metrics -> summarize_generation. Answers, JSON parsing and latency are copied.

Usage (ideal-retrieval COLIEE run):
    python3 evaluation/rescore_citations.py --dataset coliee \
        --results experiment_results/coliee_layer2/R06/layer2_generation_prompt_short.json \
        --citation-mode both \
        --output experiment_results/coliee_layer2/R06/layer2_generation_prompt_both.json
Reads <results stem>_raw_outputs.json next to --results. Runs on CPU in seconds.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))
sys.path.insert(0, str(ROOT_PATH / "src"))

from evaluation.adapters import available_datasets, load_dataset  # noqa: E402
from evaluation.dataset_defaults import resolve_dataset_paths  # noqa: E402
from evaluation.evaluate_end2end import calculate_citation_metrics  # noqa: E402
from evaluation.evaluate_generation import (  # noqa: E402
    build_documents_from_gt,
    extract_article_ids_from_citations,
    summarize_generation,
)
from RDG import ReferenceExtractor, _parse_json_output, collect_validated_citations  # noqa: E402


def rescore(per_query_results, raw_outputs, corpus, citation_mode):
    """Return per-query entries with the citation fields re-scored under citation_mode."""
    raw_by_id = {r['query_id']: r for r in raw_outputs}
    extractor = ReferenceExtractor()
    rescored = []
    for entry in per_query_results:
        qid = entry['query_id']
        if qid not in raw_by_id:
            raise ValueError(f'{qid}: no raw output to re-score')
        gt = entry['gt_citations']
        retrieved = sorted(set(gt))  # ideal retrieval, as in evaluate_generation
        valid_refs = extractor.extract_references(build_documents_from_gt(corpus, retrieved))

        citations = []
        if int(entry['json_parse_success']) == 1:
            output_dict = _parse_json_output(raw_by_id[qid]['raw_output_text'])
            if not output_dict:
                raise ValueError(f'{qid}: stored as parsable but the raw output does not parse')
            citations = extract_article_ids_from_citations(
                collect_validated_citations(output_dict, valid_refs, citation_mode=citation_mode))

        metrics = calculate_citation_metrics(citations, retrieved, gt)
        rescored.append({
            **entry,
            'llm_citations': citations,
            'hallucination_rate': metrics['citation_hallucination_rate'],
            'citation_precision': metrics['citation_precision'],
            'citation_recall': metrics['citation_recall'],
        })
    return rescored


def main(argv=None):
    parser = argparse.ArgumentParser(description='Re-score prompt-only Layer-2 outputs under another citation mode.')
    parser.add_argument('--dataset', type=str, default='coliee', choices=available_datasets())
    parser.add_argument('--corpus', type=str, default=None, help='Corpus path override (must match the original run)')
    parser.add_argument('--results', type=Path, required=True, help='Per-query results of the generation run')
    parser.add_argument('--citation-mode', type=str, required=True, choices=['short', 'both'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    with open(args.results, encoding='utf-8') as f:
        source = json.load(f)
    if source['metadata'].get('mode') != 'prompt':
        raise SystemExit(f"Only prompt-only runs can be re-scored (got mode={source['metadata'].get('mode')})")
    raw_path = args.results.with_name(f'{args.results.stem}_raw_outputs.json')
    with open(raw_path, encoding='utf-8') as f:
        raw = json.load(f)

    corpus_path, _ = resolve_dataset_paths(dataset=args.dataset, workspace_root=ROOT_PATH,
                                           provided_corpus=args.corpus, provided_queries=None)
    adapter_corpus, _, _ = load_dataset(dataset=args.dataset, workspace_root=ROOT_PATH,
                                        corpus_path=corpus_path, queries_path=None)

    per_query = rescore(source['per_query_results'], raw['raw_outputs'], adapter_corpus, args.citation_mode)
    skipped = source['aggregate_metrics'].get('skipped_queries', {}).get('queries', [])
    metadata = {
        **source['metadata'],
        'citation_mode': args.citation_mode,
        'citation_mode_requested': args.citation_mode,
        'rescored_from': str(args.results),
    }
    output = {
        'metadata': metadata,
        'aggregate_metrics': summarize_generation(per_query, skipped),
        'per_query_results': per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    # No raw-output copy: the outputs are those of the source run ('rescored_from').

    agg = output['aggregate_metrics']
    print(f"✅ Re-scored {len(per_query)} queries ({args.citation_mode}) -> {args.output}")
    print(f"Hallucination {agg['citation_hallucination_rate']['mean'] * 100:.2f}% | "
          f"Precision {agg['citation_precision']['mean']:.4f} | Recall {agg['citation_recall']['mean']:.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
