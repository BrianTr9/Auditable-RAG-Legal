"""Dataset-specific default paths for evaluators."""

from pathlib import Path
from typing import Tuple, Optional

DATASET_DEFAULTS = {
    "coliee": {
        "corpus": "benchmark/COLIEE/civil.xml",
        "queries": "benchmark/COLIEE/simple/simple_R06_jp.xml",
    },
    "cuad": {
        "corpus": "benchmark/LegalBench-RAG/corpus/cuad",
        "queries": "benchmark/LegalBench-RAG/benchmarks/cuad.json",
    },
    "housing_qa": {
        "corpus": "benchmark/housing_qa/data/statutes.tsv",
        "queries": "benchmark/housing_qa/data/questions.json",
    },
}


def get_dataset_defaults(dataset: str) -> Tuple[str, str]:
    """Get default corpus and queries paths for a dataset.
    
    Args:
        dataset: Dataset name (coliee, housing_qa, etc.)
        
    Returns:
        (corpus_path, queries_path)
    """
    if dataset not in DATASET_DEFAULTS:
        dataset = "coliee"  # Fallback to COLIEE
    
    config = DATASET_DEFAULTS[dataset]
    return config["corpus"], config["queries"]


def resolve_dataset_paths(
    dataset: str,
    workspace_root: Path,
    provided_corpus: Optional[str] = None,
    provided_queries: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve dataset corpus/queries paths intelligently.
    
    Logic:
    1. If user provides an explicit path, use it (it must exist)
    2. Otherwise use the dataset default
    3. Return (corpus_path, queries_path)
    
    Args:
        dataset: Dataset name
        workspace_root: Project root directory
        provided_corpus: Optional user-provided corpus path
        provided_queries: Optional user-provided queries path
        
    Returns:
        (corpus_path, queries_path) - absolute or relative paths
    """
    default_corpus, default_queries = get_dataset_defaults(dataset)

    # An explicitly provided path must exist: silently falling back to the default
    # would evaluate a different corpus/query set than the one requested.
    for provided in (provided_corpus, provided_queries):
        if provided:
            full = Path(provided) if Path(provided).is_absolute() else workspace_root / provided
            if not full.exists():
                raise FileNotFoundError(f"Dataset path not found: {provided}")

    return provided_corpus or default_corpus, provided_queries or default_queries
