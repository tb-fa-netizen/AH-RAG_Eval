"""
datasets/pubmedqa_loader.py
────────────────────────────────────────────────────────────────
Loads PubMedQA from HuggingFace and prepares it for RAG evaluation.

Dataset card: https://huggingface.co/datasets/qiaojin/PubMedQA
Split used  : pqa_labeled  (1,000 expert-labeled Q&A pairs)

Each sample has:
  - pubid       : PubMed article ID
  - question    : Clinical/biomedical question
  - context     : Dict with "contexts" (list of sentences) and "labels" (relevant/irrelevant)
  - long_answer : Full expert answer paragraph
  - final_decision : "yes" | "no" | "maybe"
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PubMedQASample:
    pubid: str
    question: str
    contexts: List[str]           # Abstract sentences that form the corpus
    ground_truth_contexts: List[str]  # Sentences labeled as relevant
    long_answer: str              # Expert free-text answer
    final_decision: str           # "yes" | "no" | "maybe"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def corpus_text(self) -> str:
        """Full abstract text (what we ingest into the RAG)."""
        return " ".join(self.contexts)

    @property
    def ground_truth_text(self) -> str:
        """Concatenated relevant sentences (ground-truth retrieval target)."""
        return " ".join(self.ground_truth_contexts)


def load_pubmedqa(num_samples: int = 200, seed: int = 42) -> List[PubMedQASample]:
    """
    Download and prepare PubMedQA samples.

    Parameters
    ----------
    num_samples : how many samples to return (max 1000 in pqa_labeled)
    seed        : random seed for reproducible sampling

    Returns
    -------
    List[PubMedQASample]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(f"Downloading PubMedQA (pqa_labeled), sampling {num_samples} …")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    # Deterministic shuffle + slice
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    samples = []
    for row in ds:
        ctx_dict = row["context"]
        sentences: List[str] = ctx_dict.get("contexts", [])
        labels: List[str]    = ctx_dict.get("labels", [])

        # Ground-truth = sentences labeled "relevant" (some splits use RELEVANT / IRRELEVANT)
        relevant = [
            s for s, lbl in zip(sentences, labels)
            if str(lbl).lower() in {"relevant", "1", "yes", "true"}
        ]
        # Fall back to all sentences if no label info
        if not relevant:
            relevant = sentences

        samples.append(PubMedQASample(
            pubid=str(row.get("pubid", "")),
            question=row["question"],
            contexts=sentences,
            ground_truth_contexts=relevant,
            long_answer=row.get("long_answer", ""),
            final_decision=str(row.get("final_decision", "")).lower(),
            metadata={"source": "pubmedqa", "pubid": row.get("pubid", "")},
        ))

    logger.info(f"Loaded {len(samples)} PubMedQA samples ✓")
    return samples


def pubmedqa_to_ragas_dataset(samples: List[PubMedQASample]) -> Dict[str, List]:
    """
    Convert PubMedQASample list to the dict format expected by RAGAS evaluate().

    RAGAS expects:
      {
        "question"         : List[str],
        "answer"           : List[str],   # LLM-generated (filled later)
        "contexts"         : List[List[str]],
        "ground_truth"     : List[str],
      }
    """
    return {
        "question":     [s.question          for s in samples],
        "answer":       ["" for _            in samples],   # filled after RAG run
        "contexts":     [[s.corpus_text]     for s in samples],
        "ground_truth": [s.long_answer       for s in samples],
    }
