#!/usr/bin/env python3
"""
Aggregate per-year COLIEE results (H18-R06) into the numbers reported in the paper.

Runs locally on the JSON outputs only (no model, no index).

Usage:
    # Layer 2, ideal retrieval (Table 2): per-year dirs written by slurm_scripts/job_layer2_*.sh
    python3 evaluation/aggregate_coliee.py layer2 experiment_results/coliee_layer2 \
        --compare experiment_results_generation --out experiment_results/coliee_layer2/SUMMARY.md

    # Layer 1, retrieval: per-year dirs written by slurm_scripts/job_layer1_coliee.sh
    python3 evaluation/aggregate_coliee.py layer1 experiment_results/coliee_layer1 \
        --out experiment_results/coliee_layer1/SUMMARY.md

ROOT is either a directory of per-year subdirectories (H18/, ..., R06/) or a single flat
directory holding the result files (e.g. the 2026-05 R06 run in experiment_results_generation/).

Reported subsets:
    all      every loaded query with at least one GT article
    R02-R06  exams sitting after the 2017 Civil Code reform took effect (April 2020)
    R06      the 2024 exam (after the Llama-3.1 and GPT-4o knowledge cutoffs)

Statistics (reference system: RDG for Layer 2, SPHR for Layer 1):
    - continuous per-query metrics: two-sided paired t-test (as for CUAD, Table 1)
    - binary per-query outcomes (answer correct, any fabricated ID): exact McNemar test
    - Bonferroni within each (metric, subset) family over the comparisons made
    - 95% percentile bootstrap CIs for Layer-2 means
Hallucination follows evaluate_generation.py: per-query |C \\ R| / |C| over outputs with
parsable JSON (macro). The pooled rate sum|C \\ R| / sum|C| and the number of parsable outputs
with no citation at all are reported next to it, because an empty C scores 0 in the macro mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest, ttest_rel

LAYER2_SYSTEMS = {  # file stem -> label used in Table 2
    'rdg': 'RDG (8B)',
    'prompt_short': 'Prompt-only 8B',
    'prompt_both': 'Prompt-only 8B (Lenient)',
    'gpt4o_short': 'GPT-4o',
}
LAYER1_SYSTEMS = {
    'sphr': 'SPHR',
    'flat_chunksize300': 'Flat-300',
    'flat_chunksize512': 'Flat-512',
}
K_VALUES = (3, 5, 10)
SUBSETS = {
    'all': None,
    'R02-R06': {'R02', 'R03', 'R04', 'R05', 'R06'},
    'R06': {'R06'},
}
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def year_of(query_id: str) -> str:
    return query_id.split('-', 1)[0]


def result_files(root: Path, filename: str) -> List[Path]:
    """The file directly in root, plus one per year subdirectory."""
    files = [root / filename] if (root / filename).is_file() else []
    files += sorted(p for p in root.glob(f'*/{filename}') if p.is_file())
    return files


def load_per_query(root: Path, filename: str) -> Tuple[Dict[str, dict], List[dict]]:
    """Merge per_query_results across years. Returns ({query_id: record}, skipped_queries)."""
    merged: Dict[str, dict] = {}
    skipped: List[dict] = []
    for path in result_files(root, filename):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for rec in data['per_query_results']:
            qid = rec['query_id']
            if qid in merged:
                raise ValueError(f'Duplicate query_id {qid} in {path}')
            merged[qid] = rec
        skipped += data.get('aggregate_metrics', {}).get('skipped_queries', {}).get('queries', [])
    return merged, skipped


def in_subset(qid: str, subset: Optional[set]) -> bool:
    return subset is None or year_of(qid) in subset


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired t-test. Returns (t, p); p=1 when the differences have no variance."""
    diff = a - b
    if len(diff) < 2 or np.allclose(diff.std(ddof=1), 0.0):
        return 0.0, 1.0
    res = ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> Tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes. Returns (n10, n01, p)."""
    x = x.astype(bool)
    y = y.astype(bool)
    n10 = int(np.sum(x & ~y))
    n01 = int(np.sum(~x & y))
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, float(binomtest(n10, n10 + n01, 0.5).pvalue)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    if len(values) == 0:
        return float('nan'), float('nan')
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def marker_for(p_bonf: float) -> str:
    if p_bonf < 0.01:
        return r'\sT'
    if p_bonf < 0.05:
        return r'\st'
    return '—'


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------

def layer2_query_view(rec: dict) -> dict:
    """Per-query quantities, with the stored hallucination rate re-derived as a consistency check."""
    cited = set(rec['llm_citations'])
    gt = set(rec['gt_citations'])  # ideal retrieval: R = G
    fabricated = cited - gt
    parsed = int(rec['json_parse_success']) == 1
    rate = len(fabricated) / len(cited) if cited else 0.0
    if abs(rate - float(rec['hallucination_rate'])) > 1e-9:
        raise ValueError(f"{rec['query_id']}: stored hallucination_rate {rec['hallucination_rate']} != {rate}")
    return {
        'parsed': parsed,
        'n_cited': len(cited),
        'n_fabricated': len(fabricated),
        'hallucination_rate': rate,
        'precision': float(rec['citation_precision']),
        'recall': float(rec['citation_recall']),
        'correct': rec.get('answer_correct'),
        'time': float(rec['generation_time']),
    }


def layer2_summary(views: Dict[str, dict], rng: np.random.Generator) -> dict:
    vs = list(views.values())
    parsed = [v for v in vs if v['parsed']]
    hall = np.array([v['hallucination_rate'] for v in parsed])
    prec = np.array([v['precision'] for v in vs])
    rec = np.array([v['recall'] for v in vs])
    acc = np.array([v['correct'] for v in vs if v['correct'] is not None], dtype=float)
    n_cited = sum(v['n_cited'] for v in parsed)
    n_fab = sum(v['n_fabricated'] for v in parsed)
    return {
        'n': len(vs),
        'json': np.mean([v['parsed'] for v in vs]) if vs else 0.0,
        'n_parsed': len(parsed),
        'hall_macro': hall.mean() if len(hall) else 0.0,
        'hall_macro_ci': bootstrap_ci(hall, rng),
        'hall_pooled': n_fab / n_cited if n_cited else 0.0,
        'pooled_counts': (n_fab, n_cited),
        'n_fab_queries': sum(1 for v in parsed if v['n_fabricated'] > 0),
        'n_empty_citations': sum(1 for v in parsed if v['n_cited'] == 0),
        'precision': prec.mean() if len(prec) else 0.0,
        'precision_ci': bootstrap_ci(prec, rng),
        'recall': rec.mean() if len(rec) else 0.0,
        'recall_ci': bootstrap_ci(rec, rng),
        'accuracy': acc.mean() if len(acc) else 0.0,
        'accuracy_ci': bootstrap_ci(acc, rng),
        'n_correct': int(acc.sum()),
        'latency': np.mean([v['time'] for v in vs]) if vs else 0.0,
    }


def layer2_tests(ref: Dict[str, dict], other: Dict[str, dict]) -> List[dict]:
    """RDG vs one other system on the queries both evaluated. p values are raw (Bonferroni applied by caller)."""
    common = sorted(set(ref) & set(other))
    out = []
    for metric in ('precision', 'recall'):
        a = np.array([ref[q][metric] for q in common])
        b = np.array([other[q][metric] for q in common])
        t, p = paired_ttest(a, b)
        out.append({'metric': metric, 'n': len(common), 'mean_ref': a.mean(), 'mean_other': b.mean(),
                    'test': 'paired t', 'stat': f't={t:+.3f}', 'p': p})
    correct = [q for q in common if ref[q]['correct'] is not None and other[q]['correct'] is not None]
    a = np.array([ref[q]['correct'] for q in correct])
    b = np.array([other[q]['correct'] for q in correct])
    n10, n01, p = mcnemar_exact(a, b)
    out.append({'metric': 'accuracy', 'n': len(correct), 'mean_ref': a.mean() if len(a) else 0.0,
                'mean_other': b.mean() if len(b) else 0.0, 'test': 'McNemar',
                'stat': f'RDG-only={n10}, other-only={n01}', 'p': p})
    both_parsed = [q for q in common if ref[q]['parsed'] and other[q]['parsed']]
    a = np.array([ref[q]['n_fabricated'] > 0 for q in both_parsed])
    b = np.array([other[q]['n_fabricated'] > 0 for q in both_parsed])
    n10, n01, p = mcnemar_exact(a, b)
    out.append({'metric': 'any fabricated ID', 'n': len(both_parsed), 'mean_ref': a.mean() if len(a) else 0.0,
                'mean_other': b.mean() if len(b) else 0.0, 'test': 'McNemar',
                'stat': f'RDG-only={n10}, other-only={n01}', 'p': p})
    return out


def api_models(root: Path, stem: str) -> Dict[str, int]:
    """Count of the API snapshot recorded per raw output; more than one key means the run mixed snapshots."""
    counts: Dict[str, int] = {}
    for path in result_files(root, f'layer2_generation_{stem}_raw_outputs.json'):
        with open(path, encoding='utf-8') as f:
            for out in json.load(f).get('raw_outputs', []):
                if out.get('api_model'):
                    counts[out['api_model']] = counts.get(out['api_model'], 0) + 1
    return counts


def run_layer2(root: Path, compare: Optional[Path]) -> str:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    systems: Dict[str, Dict[str, dict]] = {}
    lines = [f'# COLIEE Layer 2 (ideal retrieval) — aggregated from `{root}`', '']
    for stem in LAYER2_SYSTEMS:
        recs, skipped = load_per_query(root, f'layer2_generation_{stem}.json')
        if not recs:
            continue
        systems[stem] = {q: layer2_query_view(r) for q, r in recs.items() if r['gt_citations']}
        years = sorted({year_of(q) for q in recs})
        lines.append(f'- `{stem}`: {len(recs)} queries from {len(years)} year(s) ({years[0]}–{years[-1]}); '
                     f'skipped by the evaluator: {[s["query_id"] for s in skipped] or "none"}')
        if stem == 'gpt4o_short':
            lines.append(f'  - API snapshots that answered: {api_models(root, stem) or "not recorded (older run)"}')
    if 'rdg' not in systems:
        raise SystemExit(f'No layer2_generation_rdg.json found under {root}')
    lines.append('')

    for subset_name, subset in SUBSETS.items():
        views = {s: {q: v for q, v in qv.items() if in_subset(q, subset)} for s, qv in systems.items()}
        if not views['rdg']:
            continue
        lines += [f'## Subset {subset_name} (n = {len(views["rdg"])})', '',
                  '| System | n | Halluc. (macro) [95% CI] | Halluc. (pooled) | #q with fabricated ID | '
                  '#parsed with no citation | Prec [95% CI] | Rec [95% CI] | Acc [95% CI] | JSON | Lat. (s) |',
                  '|---|---:|---|---|---:|---:|---|---|---|---:|---:|']
        for stem, qv in views.items():
            s = layer2_summary(qv, rng)
            lines.append(
                f"| {LAYER2_SYSTEMS[stem]} | {s['n']} "
                f"| {100 * s['hall_macro']:.2f}% [{100 * s['hall_macro_ci'][0]:.2f}, {100 * s['hall_macro_ci'][1]:.2f}] "
                f"| {100 * s['hall_pooled']:.2f}% ({s['pooled_counts'][0]}/{s['pooled_counts'][1]}) "
                f"| {s['n_fab_queries']} | {s['n_empty_citations']} "
                f"| {s['precision']:.3f} [{s['precision_ci'][0]:.3f}, {s['precision_ci'][1]:.3f}] "
                f"| {s['recall']:.3f} [{s['recall_ci'][0]:.3f}, {s['recall_ci'][1]:.3f}] "
                f"| {s['accuracy']:.3f} [{s['accuracy_ci'][0]:.3f}, {s['accuracy_ci'][1]:.3f}] ({s['n_correct']}) "
                f"| {s['json']:.3f} | {s['latency']:.2f} |"
            )
        lines.append('')

        others = [s for s in views if s != 'rdg']
        if others:
            n_cmp = len(others)
            lines += [f'Tests vs RDG (Bonferroni ×{n_cmp} per metric):', '',
                      '| Metric | RDG vs | n | RDG | other | Test | Statistic | p (raw) | p_bonf | marker |',
                      '|---|---|---:|---:|---:|---|---|---:|---:|:---:|']
            rows = []
            for stem in others:
                for t in layer2_tests(views['rdg'], views[stem]):
                    rows.append((t['metric'], stem, t))
            order = ['precision', 'recall', 'accuracy', 'any fabricated ID']
            for metric, stem, t in sorted(rows, key=lambda r: (order.index(r[0]), others.index(r[1]))):
                p_bonf = min(t['p'] * n_cmp, 1.0)
                lines.append(f"| {metric} | {LAYER2_SYSTEMS[stem]} | {t['n']} | {t['mean_ref']:.3f} "
                             f"| {t['mean_other']:.3f} | {t['test']} | {t['stat']} | {t['p']:.3e} "
                             f"| {p_bonf:.3e} | {marker_for(p_bonf)} |")
            lines.append('')

    if compare is not None:
        lines += compare_layer2(root, systems, compare)
    return '\n'.join(lines)


def compare_layer2(root: Path, systems: Dict[str, Dict[str, dict]], compare: Path) -> List[str]:
    """Sanity check of the new run against an earlier run on the overlapping queries (e.g. R06)."""
    lines = [f'## Sanity check against `{compare}` (overlapping queries)', '',
             '| System | n overlap | same Y/N prediction | same cited IDs | Acc old → new | '
             'Halluc. macro old → new | Lat. old → new (s) |',
             '|---|---:|---:|---:|---|---|---|']
    for stem in LAYER2_SYSTEMS:
        old_recs, _ = load_per_query(compare, f'layer2_generation_{stem}.json')
        new_recs, _ = load_per_query(root, f'layer2_generation_{stem}.json')
        common = sorted(set(old_recs) & set(new_recs))
        if not common or stem not in systems:
            continue
        same_pred = np.mean([old_recs[q]['predicted_answer'] == new_recs[q]['predicted_answer'] for q in common])
        same_cit = np.mean([sorted(old_recs[q]['llm_citations']) == sorted(new_recs[q]['llm_citations']) for q in common])

        def acc(recs):
            return np.mean([recs[q]['answer_correct'] or 0 for q in common])

        def hall(recs):
            vals = [recs[q]['hallucination_rate'] for q in common if recs[q]['json_parse_success'] == 1]
            return 100 * np.mean(vals) if vals else 0.0

        def lat(recs):
            return np.mean([recs[q]['generation_time'] for q in common])

        lines.append(f'| {LAYER2_SYSTEMS[stem]} | {len(common)} | {same_pred:.3f} | {same_cit:.3f} '
                     f'| {acc(old_recs):.3f} → {acc(new_recs):.3f} | {hall(old_recs):.2f}% → {hall(new_recs):.2f}% '
                     f'| {lat(old_recs):.2f} → {lat(new_recs):.2f} |')
    lines.append('')
    return lines


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------

L1_METRICS = ('hit@k', 'precision@k', 'recall@k', 'mrr', 'ndcg@k')
L1_TESTED = ('precision@k', 'recall@k', 'mrr', 'ndcg@k')  # Hit@k is binary: untested, as for CUAD


def run_layer1(root: Path) -> str:
    lines = [f'# COLIEE Layer 1 (retrieval) — aggregated from `{root}`', '']
    data: Dict[Tuple[str, int], Dict[str, dict]] = {}
    for stem in LAYER1_SYSTEMS:
        for k in K_VALUES:
            recs, _ = load_per_query(root, f'layer1_retrieval_{stem}_k{k}.json')
            if recs:
                data[(stem, k)] = recs
    if not data:
        raise SystemExit(f'No layer1_retrieval_*.json found under {root}')

    some = next(iter(data.values()))
    no_gt = sorted(q for q, r in some.items() if not r['ground_truth'])
    lines += [f'- Queries loaded: {len(some)}; excluded for having no GT article: {no_gt or "none"}', '']
    for key, recs in data.items():
        if set(recs) != set(some):
            raise ValueError(f'{key}: query set differs from the other result files')

    systems = [s for s in LAYER1_SYSTEMS if any((s, k) in data for k in K_VALUES)]
    for subset_name, subset in SUBSETS.items():
        qids = sorted(q for q in some if some[q]['ground_truth'] and in_subset(q, subset))
        if not qids:
            continue
        lines += [f'## Subset {subset_name} (n = {len(qids)})', '',
                  '| System | k | ' + ' | '.join(L1_METRICS) + ' |',
                  '|---|---:|' + '---:|' * len(L1_METRICS)]
        for stem in systems:
            for k in K_VALUES:
                if (stem, k) not in data:
                    continue
                vals = [np.mean([data[(stem, k)][q]['metrics'][m] for q in qids]) for m in L1_METRICS]
                lines.append(f'| {LAYER1_SYSTEMS[stem]} | {k} | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
        lines.append('')

        pairs = [(a, b) for i, a in enumerate(systems) for b in systems[i + 1:]]
        if pairs:
            lines += [f'Paired t-tests (Bonferroni ×{len(pairs)} per (metric, k)):', '',
                      '| Metric | k | A vs B | mean A | mean B | Δ | t | p_bonf | marker |',
                      '|---|---:|---|---:|---:|---:|---:|---:|:---:|']
            for m in L1_TESTED:
                for k in K_VALUES:
                    for a_s, b_s in pairs:
                        if (a_s, k) not in data or (b_s, k) not in data:
                            continue
                        a = np.array([data[(a_s, k)][q]['metrics'][m] for q in qids])
                        b = np.array([data[(b_s, k)][q]['metrics'][m] for q in qids])
                        t, p = paired_ttest(a, b)
                        p_bonf = min(p * len(pairs), 1.0)
                        lines.append(f'| {m} | {k} | {LAYER1_SYSTEMS[a_s]} vs {LAYER1_SYSTEMS[b_s]} '
                                     f'| {a.mean():.4f} | {b.mean():.4f} | {a.mean() - b.mean():+.4f} '
                                     f'| {t:+.3f} | {p_bonf:.3e} | {marker_for(p_bonf)} |')
            lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Aggregate per-year COLIEE results.')
    parser.add_argument('layer', choices=['layer1', 'layer2'])
    parser.add_argument('root', type=Path, help='Results root (per-year subdirectories or one flat directory)')
    parser.add_argument('--compare', type=Path, default=None,
                        help='Layer 2 only: earlier run to compare on overlapping queries (sanity check)')
    parser.add_argument('--out', type=Path, default=None, help='Write the Markdown report here as well')
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        raise SystemExit(f'Not a directory: {args.root}')
    report = run_layer2(args.root, args.compare) if args.layer == 'layer2' else run_layer1(args.root)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
