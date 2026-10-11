"""COLIEE dataset adapter."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Corpus, DatasetAdapter, Query, resolve_path


class ColieeAdapter(DatasetAdapter):
    name = "coliee"

    def load_corpus(self, workspace_root: Path, corpus_path: Optional[str] = None) -> Corpus:
        path = resolve_path(workspace_root, corpus_path, "benchmark/COLIEE/civil.xml")
        tree = ET.parse(path)
        root = tree.getroot()

        corpus: Corpus = {}
        for article in root.findall("Article"):
            article_num = article.get("num")
            caption_elem = article.find("caption")
            caption = caption_elem.text.strip() if caption_elem is not None and caption_elem.text else ""
            text_elem = article.find("text")
            article_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
            if article_num and article_text:
                corpus[article_num] = {
                    "caption": caption or article_num,
                    "text": article_text,
                }
        return corpus

    def load_queries(self, workspace_root: Path, queries_path: Optional[str] = None) -> List[Query]:
        path = resolve_path(workspace_root, queries_path, "benchmark/COLIEE/simple/simple_R06_jp.xml")
        tree = ET.parse(path)
        root = tree.getroot()

        queries: List[Query] = []
        for pair in root.findall("pair"):
            pair_id = pair.get("id")
            ground_truth = (pair.get("label") or "").strip() or None
            t2_elem = pair.find("t2")
            question = "".join(t2_elem.itertext()).strip() if t2_elem is not None else ""

            relevant_articles: List[str] = []
            for art in pair.findall("article"):
                val = ("".join(art.itertext()) if art is not None else "").strip()
                if val:
                    relevant_articles.append(val)

            queries.append(
                {
                    "query_id": pair_id or "",
                    "question": question,
                    "ground_truth_answer": ground_truth,
                    "relevant_articles": relevant_articles,
                }
            )
        return queries
