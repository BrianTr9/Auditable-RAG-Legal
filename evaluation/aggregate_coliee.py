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
Further subsets, added after the 2026-10 re-run: H18-R05 (everything except R06, which was also used while the
prompt and the scoring were developed) and H18-R01 (exams before the reform took effect).

Extra report sections (after the per-subset tables, so the per-subset layout is unchanged):
    Layer 2: answer behaviour against the majority-class baseline (per-class recall, balanced accuracy), citation
             recall by number of gold articles, composition of the invalid IDs of the prompt-only runs, and a
             cluster bootstrap over exam questions for the headline contrasts.
    Layer 1: precision and NDCG per retrieval slot (P@k = hits / k, ideal DCG over min(|gold|, k)) next to the
             per-distinct-article numbers of the main tables, and a cluster bootstrap of SPHR minus each flat baseline.
The 1216 statements come from 407 exam questions (up to five statements share one question and its gold articles),
so the bootstrap resamples whole questions; the naive per-statement interval is printed next to it.

Hallucination follows evaluate_generation.py: per-query |C \\ R| / |C| over outputs with
parsable JSON (macro). The pooled rate sum|C \\ R| / sum|C| and the number of parsable outputs
with no citation at all are reported next to it, because an empty C scores 0 in the macro mean.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import re
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
    'H18-R05': {f'H{n}' for n in range(18, 31)} | {'R01', 'R02', 'R03', 'R04', 'R05'},   # R06 held out
    'H18-R01': {f'H{n}' for n in range(18, 31)} | {'R01'},                                # before the reform
}
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def year_of(query_id: str) -> str:
    """Exam year of a query id, normalised: simple_R01_jp.xml mixes the prefixes 'R01-' and 'R1-'."""
    head = query_id.split('-', 1)[0]
    m = re.fullmatch(r'([HR])0*(\d+)', head)
    return f'{m.group(1)}{int(m.group(2)):02d}' if m else head


def question_of(query_id: str) -> str:
    """Exam question a statement belongs to (statements A/E/I/O/U of one question share their gold articles)."""
    return f'{year_of(query_id)}-{query_id.split("-")[1]}' if query_id.count('-') >= 2 else query_id


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

INVALID_PREFIX = '__INVALID__::'


def decode_invalid(token: str) -> str:
    """evaluate_generation keeps an invalid citation as '__INVALID__::<hex of the text>'; return the text."""
    try:
        return binascii.unhexlify(token[len(INVALID_PREFIX):]).decode('utf-8', 'replace')
    except (binascii.Error, ValueError):
        return token


def invalid_kind(text: str) -> str:
    if text == '__EMPTY_CITATION__':
        return 'empty citation'
    if text.startswith('doc_'):
        return 'copy of the prompt example (doc_N...)'
    if '_Section_' in text and ('[' in text or ' ' in text):
        return 'full form with extra text'
    if '_Section_' in text:
        return 'full form of a real article'
    return 'other'


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
        'qid': rec['query_id'],
        'label': rec.get('ground_truth_answer'),
        'pred': rec.get('predicted_answer'),
        'n_gold': len(gt),
        'n_fab_flag': 1.0 if fabricated else 0.0,
        'invalid_tokens': [decode_invalid(t) for t in rec['llm_citations'] if t.startswith(INVALID_PREFIX)],
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


# ---------------------------------------------------------------------------
# Extra analyses (answer behaviour, recall by gold count, invalid IDs, cluster bootstrap)
# ---------------------------------------------------------------------------

def cluster_bootstrap(values: Dict[str, float], rng: np.random.Generator):
    """Mean of per-statement values with a 95% percentile interval that resamples whole exam questions.

    Statements of one question share their gold articles and are not independent, so the interval resamples
    questions; the naive per-statement interval is returned next to it. Returns
    (estimate, cluster CI, naive CI, number of questions)."""
    qids = sorted(values)
    if not qids:
        nan = float('nan')
        return nan, (nan, nan), (nan, nan), 0
    questions = sorted({question_of(q) for q in qids})
    index = {q: i for i, q in enumerate(questions)}
    total = np.zeros(len(questions))
    count = np.zeros(len(questions))
    for q in qids:
        i = index[question_of(q)]
        total[i] += values[q]
        count[i] += 1
    est = float(total.sum() / count.sum())
    pick = rng.integers(0, len(questions), size=(N_BOOTSTRAP, len(questions)))
    cluster_means = total[pick].sum(axis=1) / np.maximum(count[pick].sum(axis=1), 1)
    arr = np.array([values[q] for q in qids])
    naive_means = arr[rng.integers(0, len(arr), size=(N_BOOTSTRAP, len(arr)))].mean(axis=1)
    return (est, tuple(float(x) for x in np.percentile(cluster_means, [2.5, 97.5])),
            tuple(float(x) for x in np.percentile(naive_means, [2.5, 97.5])), len(questions))


def fmt_ci(ci) -> str:
    return f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'


def answer_behaviour(views: Dict[str, dict]) -> Optional[dict]:
    """Y/N behaviour against the majority-class baseline (always answering the more frequent gold label)."""
    vs = [v for v in views.values() if v['label'] in ('Y', 'N')]
    if not vs:
        return None
    labels = [v['label'] for v in vs]
    majority = max(labels.count('Y'), labels.count('N')) / len(vs)
    correct = sum(1 for v in vs if v['correct'])
    gold_y = [v for v in vs if v['label'] == 'Y']
    gold_n = [v for v in vs if v['label'] == 'N']
    rec_y = float(np.mean([v['pred'] == 'Y' for v in gold_y])) if gold_y else float('nan')
    rec_n = float(np.mean([v['pred'] == 'N' for v in gold_n])) if gold_n else float('nan')
    return {
        'n': len(vs), 'acc': correct / len(vs), 'majority': majority,
        'p_vs_majority': float(binomtest(correct, len(vs), majority, alternative='greater').pvalue),
        'says_y': float(np.mean([v['pred'] == 'Y' for v in vs])),
        'recall_y': rec_y, 'recall_n': rec_n,
        'balanced': float(np.nanmean([rec_y, rec_n])) if (gold_y or gold_n) else float('nan'),
        'unparsed': sum(1 for v in vs if not v['parsed']),
    }


def gold_bucket(n_gold: int) -> str:
    return '1' if n_gold <= 1 else ('2' if n_gold == 2 else '3+')


def layer2_contrasts(V: Dict[str, Dict[str, dict]]) -> List[Tuple[str, Dict[str, float]]]:
    """Headline contrasts as per-statement difference vectors (only those whose systems are present)."""
    def diff(a, b, f, both_parsed=False):
        if a not in V or b not in V:
            return None
        ids = [q for q in V[a] if q in V[b] and (not both_parsed or (V[a][q]['parsed'] and V[b][q]['parsed']))]
        return {q: f(V[a][q]) - f(V[b][q]) for q in ids} or None

    acc = lambda v: float(v['correct'] or 0)
    out = []
    for name, d in (
        ('fabricated-ID rate: prompt-only (strict) minus RDG', diff('prompt_short', 'rdg', lambda v: v['n_fab_flag'], True)),
        ('fabricated-ID rate: prompt-only (Lenient) minus RDG', diff('prompt_both', 'rdg', lambda v: v['n_fab_flag'], True)),
        ('citation recall: RDG minus GPT-4o', diff('rdg', 'gpt4o_short', lambda v: v['recall'])),
        ('citation precision: RDG minus GPT-4o', diff('rdg', 'gpt4o_short', lambda v: v['precision'])),
        ('citation recall: RDG minus prompt-only (Lenient)', diff('rdg', 'prompt_both', lambda v: v['recall'])),
        ('accuracy: RDG minus prompt-only', diff('rdg', 'prompt_short', acc)),
        ('accuracy: GPT-4o minus RDG', diff('gpt4o_short', 'rdg', acc)),
    ):
        if d:
            out.append((name, d))
    ab = answer_behaviour(V['rdg']) if 'rdg' in V else None
    if ab:
        out.append(('accuracy: RDG minus the majority-class baseline',
                    {q: acc(v) - ab['majority'] for q, v in V['rdg'].items() if v['label'] in ('Y', 'N')}))
    return out


def layer2_extra_sections(systems: Dict[str, Dict[str, dict]]) -> List[str]:
    lines: List[str] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    subset_views = {name: {s: {q: v for q, v in qv.items() if in_subset(q, sub)} for s, qv in systems.items()}
                    for name, sub in SUBSETS.items()}

    lines += ['## Answer behaviour against the majority-class baseline', '',
              'Baseline: always answer the more frequent gold label of the subset. "says Y" is the share of statements '
              'answered Y; unparsable outputs count as wrong. p: exact binomial test of accuracy above the baseline.', '',
              '| Subset | System | n | Acc | Baseline | p (acc > baseline) | says Y | recall Y | recall N | balanced acc | #unparsable |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, V in subset_views.items():
        for stem in ('rdg', 'prompt_short', 'gpt4o_short'):
            ab = answer_behaviour(V.get(stem, {}))
            if ab:
                lines.append(f"| {name} | {LAYER2_SYSTEMS[stem]} | {ab['n']} | {ab['acc']:.3f} | {ab['majority']:.3f} "
                             f"| {ab['p_vs_majority']:.2e} | {ab['says_y']:.2f} | {ab['recall_y']:.2f} | {ab['recall_n']:.2f} "
                             f"| {ab['balanced']:.3f} | {ab['unparsed']} |")
    lines.append('')

    V = subset_views['all']
    present = [s for s in ('rdg', 'prompt_short', 'prompt_both', 'gpt4o_short') if V.get(s)]
    if present and 'rdg' in V:
        lines += ['## Citation recall by number of gold articles (subset all)', '',
                  '| Gold articles | n | ' + ' | '.join(LAYER2_SYSTEMS[s] for s in present) + ' |',
                  '|---|---:|' + '---:|' * len(present)]
        for bucket in ('1', '2', '3+'):
            ids = [q for q, v in V['rdg'].items() if gold_bucket(v['n_gold']) == bucket]
            if ids:
                cells = [f"{np.mean([V[s][q]['recall'] for q in ids if q in V[s]]):.3f}" for s in present]
                lines.append(f'| {bucket} | {len(ids)} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    comp = {}
    for stem in ('prompt_short', 'prompt_both'):
        if V.get(stem):
            c = collections_counter(invalid_kind(t) for v in V[stem].values() for t in v['invalid_tokens'])
            comp[stem] = (c, sum(v['n_cited'] for v in V[stem].values() if v['parsed']))
    if comp:
        kinds = sorted({k for c, _ in comp.values() for k in c})
        lines += ['## Invalid citation IDs of the prompt-only runs (subset all)', '',
                  'Strict accepts only the short ID; Lenient also accepts the full form. What remains under Lenient is '
                  'text that is not an ID of the retrieved set.', '',
                  '| Kind | ' + ' | '.join(LAYER2_SYSTEMS[s] for s in comp) + ' |', '|---|' + '---:|' * len(comp)]
        for k in kinds:
            lines.append(f'| {k} | ' + ' | '.join(str(comp[s][0].get(k, 0)) for s in comp) + ' |')
        lines.append('| total invalid / total cited | ' + ' | '.join(f'{sum(comp[s][0].values())} / {comp[s][1]}' for s in comp) + ' |')
        lines.append('')

    lines += ['## Sensitivity: cluster bootstrap over exam questions', '',
              f'{N_BOOTSTRAP} resamples of whole exam questions; the naive interval resamples statements. Differences are '
              'per statement (positive: the first system is higher).', '',
              '| Subset | Contrast | n | questions | Estimate | 95% CI (questions) | 95% CI (naive) |', '|---|---|---:|---:|---:|---|---|']
    for name in ('all', 'H18-R05', 'R06'):
        for label, values in layer2_contrasts(subset_views[name]):
            est, cl, nv, nq = cluster_bootstrap(values, rng)
            lines.append(f'| {name} | {label} | {len(values)} | {nq} | {est:+.4f} | {fmt_ci(cl)} | {fmt_ci(nv)} |')
    lines.append('')
    return lines


def collections_counter(items):
    counts: Dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return counts


def l1_standard(retrieved: List[str], gold: List[str], k: int) -> Dict[str, float]:
    """Slot-based retrieval metrics on the first k distinct articles returned.

    P@k = hits / k: every one of the k retrieval slots counts, so a second chunk of an already returned article
    uses a slot without adding a hit. The stored precision instead divides by the number of distinct articles
    returned, which measures how clean the set of provisions is and does not charge for duplicate chunks of a
    relevant article. Both are legitimate and answer different questions; NDCG's ideal ranking here holds
    min(|gold|, k) articles."""
    gold_set = set(gold)
    top = list(dict.fromkeys(retrieved))[:k]
    hits = [d in gold_set for d in top]
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return {'p': sum(hits) / k, 'r': sum(hits) / len(gold_set) if gold_set else 0.0,
            'mrr': next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0), 'ndcg': dcg / idcg if idcg else 0.0}


def layer1_extra_sections(data: Dict[Tuple[str, int], Dict[str, dict]], systems: List[str], some: Dict[str, dict]) -> List[str]:
    lines: List[str] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    std = {key: {q: l1_standard(r['retrieved'], r['ground_truth'], key[1]) for q, r in recs.items() if r['ground_truth']}
           for key, recs in data.items()}
    lines += ['## Slot-based definitions (P@k = hits / k, first k distinct articles, ideal DCG over min(|gold|, k))', '',
              'The tables above divide precision by the number of distinct articles a system returned: several chunks of one '
              'relevant article count once, so this measures how clean the set of provisions shown to the generator is. '
              'The definition below divides by the k retrieval slots, so a duplicate chunk uses a slot without adding a hit. '
              'Neither is wrong; they differ on precision only. "returned" is the mean number of distinct articles.', '',
              '| Subset | System | k | returned | P@k | R@k | MRR | NDCG@k |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, sub in SUBSETS.items():
        qids = sorted(q for q in some if some[q]['ground_truth'] and in_subset(q, sub))
        if not qids:
            continue
        for stem in systems:
            for k in K_VALUES:
                if (stem, k) not in data:
                    continue
                m = std[(stem, k)]
                ret = np.mean([len(set(data[(stem, k)][q]['retrieved'])) for q in qids])
                lines.append(f"| {name} | {LAYER1_SYSTEMS[stem]} | {k} | {ret:.2f} | "
                             + ' | '.join(f"{np.mean([m[q][key] for q in qids]):.4f}" for key in ('p', 'r', 'mrr', 'ndcg')) + ' |')
    lines.append('')
    lines += ['## Sensitivity: cluster bootstrap over exam questions (slot-based definitions, subset all)', '',
              f'{N_BOOTSTRAP} resamples of whole exam questions; the naive interval resamples statements.', '',
              '| k | Contrast | Metric | questions | Estimate | 95% CI (questions) | 95% CI (naive) |', '|---:|---|---|---:|---:|---|---|']
    qids = sorted(q for q in some if some[q]['ground_truth'])
    for k in K_VALUES:
        for other in ('flat_chunksize300', 'flat_chunksize512'):
            if ('sphr', k) not in data or (other, k) not in data or not qids:
                continue
            for key, label in (('p', 'P@k (hits / k)'), ('r', 'R@k'), ('mrr', 'MRR'), ('ndcg', 'NDCG@k')):
                values = {q: std[('sphr', k)][q][key] - std[(other, k)][q][key] for q in qids}
                est, cl, nv, nq = cluster_bootstrap(values, rng)
                lines.append(f'| {k} | SPHR minus {LAYER1_SYSTEMS[other]} | {label} | {nq} | {est:+.4f} | {fmt_ci(cl)} | {fmt_ci(nv)} |')
    lines.append('')
    return lines


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

    lines += layer2_extra_sections(systems)
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
    lines += layer1_extra_sections(data, systems, some)
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
