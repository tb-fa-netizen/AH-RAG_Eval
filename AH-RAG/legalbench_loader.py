"""
legalbench_loader.py
────────────────────────────────────────────────────────────────
Loads legal QA samples from a local CUAD JSON file (SQuAD format).

CUAD (Contract Understanding Atticus Dataset) is what LegalBench-RAG
was built from; it has the same char-level answer-span structure.

JSON layout
───────────
{
  "data": [
    {
      "title": "...",
      "paragraphs": [
        {
          "context": "...",          ← full contract text
          "qas": [
            {
              "question": "...",
              "answers": [{"text": "...", "answer_start": N}],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class LegalSample:
    query_id: str
    query: str
    answer: str                  # Expected answer text
    document_id: str             # Contract title
    document_text: str           # Full contract text (corpus for RAG)
    relevant_span: str           # Ground-truth char span
    char_start: int
    char_end: int
    metadata: Dict = field(default_factory=dict)

    @property
    def ground_truth_context(self) -> str:
        return self.relevant_span


def load_legalbench_rag(
    cuad_json_path: str = "/tmp/CUAD_v1.json",
    num_samples: int = 150,
    seed: int = 42,
    # backward-compat kwargs — ignored
    hf_dataset: str = "",
    local_path: str = "",
    github_url: str = "",
    use_mini: bool = True,
) -> List[LegalSample]:
    """
    Load legal QA samples from a local CUAD SQuAD-format JSON file.

    Parameters
    ----------
    cuad_json_path : path to CUAD_v1.json (or any SQuAD-format legal JSON)
    num_samples    : how many answerable QA pairs to return
    seed           : reproducible shuffle
    """
    p = Path(cuad_json_path)
    if not p.exists():
        raise FileNotFoundError(
            f"CUAD JSON not found at {cuad_json_path}.\n"
            "Download it with:\n"
            '  wget -O /tmp/CUAD_v1.json '
            '"https://zenodo.org/record/4775893/files/CUAD_v1.json"'
        )

    logger.info(f"Loading CUAD from {cuad_json_path} …")
    with open(p) as f:
        cuad = json.load(f)

    # Flatten all answerable QA pairs across all titles / paragraphs
    rows: List[dict] = []
    for article in cuad["data"]:
        title = article.get("title", "unknown")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            if not context.strip():
                continue
            for qa in para.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                rows.append({
                    "id":       qa.get("id", ""),
                    "title":    title,
                    "context":  context,
                    "question": qa["question"],
                    "answer_text":  answers[0]["text"],
                    "answer_start": int(answers[0]["answer_start"]),
                })

    logger.info(f"Found {len(rows)} answerable QA pairs in CUAD")

    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:num_samples]

    samples: List[LegalSample] = []
    for i, row in enumerate(selected):
        answer_text = row["answer_text"]
        char_start  = row["answer_start"]
        char_end    = char_start + len(answer_text)
        context     = row["context"]

        # Guard against off-by-one in rare malformed entries
        extracted = context[char_start:char_end] if char_end <= len(context) else answer_text

        samples.append(LegalSample(
            query_id=row["id"] or str(i),
            query=row["question"],
            answer=answer_text,
            document_id=row["title"],
            document_text=context,
            relevant_span=extracted,
            char_start=char_start,
            char_end=char_end,
            metadata={"source": "cuad", "title": row["title"]},
        ))

    logger.info(f"Loaded {len(samples)} CUAD legal samples ✓")
    return samples


def legalbench_to_ragas_dataset(samples: List[LegalSample]) -> Dict[str, List]:
    """Convert to RAGAS-compatible dict."""
    return {
        "question":     [s.query               for s in samples],
        "answer":       ["" for _              in samples],
        "contexts":     [[s.document_text]     for s in samples],
        "ground_truth": [s.ground_truth_context for s in samples],
    }
