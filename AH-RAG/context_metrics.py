"""
metrics/context_metrics.py
────────────────────────────────────────────────────────────────
Implements Context Precision and Context Recall — the two primary
retrieval metrics used in the paper evaluation.

Two modes
─────────
1. LLM-judged (RAGAS)  — uses a local LLM to determine relevance.
   Pros: nuanced, domain-aware.  Cons: slow, LLM availability required.

2. Token-overlap (no LLM) — fast, deterministic, no LLM needed.
   Uses F1 of token overlap between retrieved chunk and ground truth.
   Suitable as a fast fallback or sanity check.

Definitions
───────────
Context Precision@K
    Among the K retrieved chunks, what fraction are relevant?
    CP@K = (# relevant retrieved chunks up to K) / K
    Weighted variant (RAGAS): gives more credit to relevant chunks
    appearing earlier in the ranked list.

Context Recall
    Of all the information in the ground truth, what fraction
    is covered by the retrieved chunks?
    CR = (# ground-truth sentences/tokens covered) / (# total ground-truth sentences/tokens)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import nltk
import numpy as np

logger = logging.getLogger(__name__)

# Pre-download NLTK tokenizer data silently on first import
for _nltk_pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_nltk_pkg}")
    except LookupError:
        nltk.download(_nltk_pkg, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase word-tokenize (no stopword removal — keep medical/legal terms)."""
    return re.findall(r"\b\w+\b", text.lower())


def _token_overlap_f1(pred: str, ref: str) -> float:
    """
    Compute token-level F1 between pred and ref.
    Same formula as SQuAD evaluation.
    """
    pred_tokens = set(_tokenize(pred))
    ref_tokens  = set(_tokenize(ref))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _char_overlap(pred: str, ref: str) -> float:
    """Character-level overlap ratio (used in LegalBench-RAG evaluation)."""
    pred_chars = set(pred.lower())
    ref_chars  = set(ref.lower())
    if not ref_chars:
        return 0.0
    return len(pred_chars & ref_chars) / len(ref_chars)


def _is_relevant_overlap(chunk: str, ground_truth: str, threshold: float = 0.20) -> bool:
    """
    Determine if a retrieved chunk is relevant to the ground truth
    via token-overlap F1.
    """
    return _token_overlap_f1(chunk, ground_truth) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Context Precision
# ─────────────────────────────────────────────────────────────────────────────

def context_precision_at_k(
    retrieved_chunks: List[str],
    ground_truth: str,
    k: int = 5,
    relevance_threshold: float = 0.20,
    weighted: bool = True,
) -> float:
    """
    Compute Context Precision@K (token-overlap based).

    Parameters
    ----------
    retrieved_chunks    : ordered list of retrieved context strings
    ground_truth        : reference answer / relevant passage
    k                   : cutoff rank
    relevance_threshold : min token-F1 to consider a chunk relevant
    weighted            : if True, use reciprocal-rank-weighted precision (RAGAS style)

    Returns
    -------
    float in [0, 1]
    """
    if not retrieved_chunks or not ground_truth.strip():
        return 0.0

    chunks_at_k = retrieved_chunks[:k]
    relevance_flags = [
        _is_relevant_overlap(c, ground_truth, relevance_threshold)
        for c in chunks_at_k
    ]

    if not any(relevance_flags):
        return 0.0

    if weighted:
        # RAGAS weighted CP: sum(precision@i * rel_i) / num_relevant
        score = 0.0
        num_relevant = 0
        for i, rel in enumerate(relevance_flags, start=1):
            if rel:
                num_relevant += 1
                precision_at_i = num_relevant / i
                score += precision_at_i
        total_relevant = sum(relevance_flags)
        return score / total_relevant if total_relevant else 0.0
    else:
        return sum(relevance_flags) / len(chunks_at_k)


def context_precision_curve(
    retrieved_chunks: List[str],
    ground_truth: str,
    k_values: List[int] = [1, 3, 5, 10],
    **kwargs,
) -> dict:
    """Compute CP at multiple cutoffs."""
    return {
        f"CP@{k}": context_precision_at_k(retrieved_chunks, ground_truth, k=k, **kwargs)
        for k in k_values
        if k <= len(retrieved_chunks)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context Recall
# ─────────────────────────────────────────────────────────────────────────────

def context_recall_token(
    retrieved_chunks: List[str],
    ground_truth: str,
) -> float:
    """
    Token-level Context Recall.

    Measures: what fraction of ground-truth tokens appear in the
    union of retrieved chunks?

    CR = |tokens(GT) ∩ tokens(union_of_retrieved)| / |tokens(GT)|
    """
    if not retrieved_chunks or not ground_truth.strip():
        return 0.0

    gt_tokens = set(_tokenize(ground_truth))
    if not gt_tokens:
        return 0.0

    retrieved_text = " ".join(retrieved_chunks)
    retrieved_tokens = set(_tokenize(retrieved_text))

    covered = gt_tokens & retrieved_tokens
    return len(covered) / len(gt_tokens)


def context_recall_sentence(
    retrieved_chunks: List[str],
    ground_truth: str,
    coverage_threshold: float = 0.50,
) -> float:
    """
    Sentence-level Context Recall (RAGAS style).

    Each ground-truth sentence is considered "covered" if at least
    one retrieved chunk has token-F1 ≥ coverage_threshold with it.

    CR = # covered GT sentences / total GT sentences
    """
    sentences = nltk.sent_tokenize(ground_truth)

    if not sentences:
        return 0.0

    covered = 0
    for sent in sentences:
        if any(
            _token_overlap_f1(chunk, sent) >= coverage_threshold
            for chunk in retrieved_chunks
        ):
            covered += 1

    return covered / len(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# Legal-specific: character-overlap precision (LegalBench-RAG metric)
# ─────────────────────────────────────────────────────────────────────────────

def legal_char_precision(
    retrieved_chunks: List[str],
    relevant_span: str,
    char_overlap_threshold: float = 0.50,
) -> float:
    """
    LegalBench-RAG style precision: a retrieved chunk is relevant
    iff it has ≥ char_overlap_threshold character overlap with the
    ground-truth legal span.
    """
    if not retrieved_chunks or not relevant_span.strip():
        return 0.0

    relevant_count = sum(
        1 for chunk in retrieved_chunks
        if _char_overlap(chunk, relevant_span) >= char_overlap_threshold
    )
    return relevant_count / len(retrieved_chunks)


def legal_char_recall(
    retrieved_chunks: List[str],
    relevant_span: str,
) -> float:
    """
    Char-level recall: what fraction of the relevant span's characters
    appear in the retrieved chunks?
    """
    if not relevant_span.strip():
        return 0.0
    retrieved_text = " ".join(retrieved_chunks)
    return _char_overlap(retrieved_text, relevant_span)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate scorer — convenience function
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_chunks: List[str],
    ground_truth: str,
    domain: str = "general",             # "medical" | "legal" | "general"
    k_values: Optional[List[int]] = None,
) -> dict:
    """
    Compute the full set of retrieval metrics for one sample.

    Returns a flat dict of metric_name → float.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {}

    # Context Precision (weighted, RAGAS-style)
    for k in k_values:
        if k <= len(retrieved_chunks):
            results[f"context_precision_at_{k}"] = context_precision_at_k(
                retrieved_chunks, ground_truth, k=k, weighted=True
            )

    # Context Recall (two variants)
    results["context_recall_token"]    = context_recall_token(retrieved_chunks, ground_truth)
    results["context_recall_sentence"] = context_recall_sentence(retrieved_chunks, ground_truth)

    # Domain-specific
    if domain == "legal":
        results["legal_char_precision"] = legal_char_precision(retrieved_chunks, ground_truth)
        results["legal_char_recall"]    = legal_char_recall(retrieved_chunks, ground_truth)

    # MRR (Mean Reciprocal Rank) — useful for legal single-span evaluation
    results["mrr"] = _mean_reciprocal_rank(retrieved_chunks, ground_truth)

    return results


def _mean_reciprocal_rank(retrieved_chunks: List[str], ground_truth: str, threshold: float = 0.20) -> float:
    """MRR: reciprocal of rank of first relevant chunk."""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if _is_relevant_overlap(chunk, ground_truth, threshold):
            return 1.0 / i
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS wrapper (LLM-judged, requires working Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ragas_metrics(
    dataset_dict: dict,
    llm,
    embeddings,
    metrics_to_run: Optional[List[str]] = None,
) -> dict:
    """
    Run RAGAS evaluation using local LLM + embeddings.

    Parameters
    ----------
    dataset_dict   : {"question": [...], "answer": [...],
                      "contexts": [[...], ...], "ground_truth": [...]}
    llm            : LlamaIndex LLM (e.g. Ollama instance)
    embeddings     : LlamaIndex embedding model
    metrics_to_run : subset of ["context_precision", "context_recall",
                                "faithfulness", "answer_relevancy"]

    Returns
    -------
    dict of metric → mean score
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        )
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install ragas datasets")

    if metrics_to_run is None:
        metrics_to_run = ["context_precision", "context_recall",
                          "faithfulness", "answer_relevancy"]

    metric_map = {
        "context_precision":  context_precision,
        "context_recall":     context_recall,
        "faithfulness":       faithfulness,
        "answer_relevancy":   answer_relevancy,
    }
    selected = [metric_map[m] for m in metrics_to_run if m in metric_map]

    # Wrap local models for RAGAS
    ragas_llm   = LlamaIndexLLMWrapper(llm)
    ragas_embed = LlamaIndexEmbeddingsWrapper(embeddings)
    for m in selected:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embed

    hf_dataset = Dataset.from_dict(dataset_dict)
    result = evaluate(hf_dataset, metrics=selected)

    return dict(result)
