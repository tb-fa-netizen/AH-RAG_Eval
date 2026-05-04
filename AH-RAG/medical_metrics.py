"""
metrics/medical_metrics.py
────────────────────────────────────────────────────────────────
Medical-domain evaluation metrics for the RAG system.

Metrics implemented
───────────────────
1. BioScore (BERTScore with BioBERT)
   ─ Measures semantic similarity between generated answer and
     ground-truth using a domain-tuned BERT model.
   ─ Captures whether the answer is *factually aligned*, not just
     keyword-matched.

2. Medical Terminology Coverage (MTC)
   ─ Uses scispaCy NER to extract UMLS entities from both the
     ground-truth and the answer.
   ─ Reports: entity precision, recall, F1.
   ─ Critical for evaluating whether the system uses correct medical
     terminology (drug names, conditions, procedures).

3. Answer Accuracy (PubMedQA decision match)
   ─ For yes/no/maybe questions, checks if the generated answer
     matches the expert final_decision.

4. Retrieval Coverage for Medical Terms
   ─ Checks what fraction of ground-truth medical entities appear
     in the retrieved context (not just the answer).

"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BioScore (BERTScore with BioBERT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bioscores(
    predictions: List[str],
    references: List[str],
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    
    """Compute BERTScore using BioBERT as the backbone (more sensitive to
    medical terminology than vanilla BERT).

    Returns
    -------
    dict with keys "precision", "recall", "f1" — each a list of floats.
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        raise ImportError("Run: pip install bert-score")

    logger.info(f"Computing BioScore with {model_name} on {len(predictions)} samples …")
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_name,
        batch_size=batch_size,
        device=device,
        lang="en",
        verbose=False,
    )
    return {
        "bioscore_precision": P.tolist(),
        "bioscore_recall":    R.tolist(),
        "bioscore_f1":        F1.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Medical Terminology Coverage (scispaCy NER)
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None  # lazy-load scispaCy


def _get_scispacy_nlp(model: str = "en_core_sci_lg"):
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(model)
        except OSError:
            logger.warning(
                f"scispaCy model '{model}' not found. "
                "Run: pip install scispacy && python -m spacy download en_core_sci_lg\n"
                "Falling back to regex-based entity extraction."
            )
            _nlp = None
    return _nlp


def _extract_entities_scispacy(text: str, model: str = "en_core_sci_lg") -> List[str]:
    """Extract medical entities using scispaCy."""
    nlp = _get_scispacy_nlp(model)
    if nlp is None:
        return _extract_entities_regex(text)
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]


def _extract_entities_regex(text: str) -> List[str]:
    """
    Fallback: extract capitalised noun phrases that look like
    medical terms (not perfect, but fast and dependency-free).
    """
    # Match 1-3 capitalised words, optionally preceded by "mg", "%", numbers
    pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
    return [m.lower() for m in re.findall(pattern, text)]


def medical_terminology_f1(
    prediction: str,
    reference: str,
    scispacy_model: str = "en_core_sci_lg",
) -> Dict[str, float]:
    """
    Medical entity-level precision, recall, F1.

    Treats each unique medical entity as a token; computes micro-averaged F1.
    """
    pred_ents = set(_extract_entities_scispacy(prediction, scispacy_model))
    ref_ents  = set(_extract_entities_scispacy(reference,  scispacy_model))

    if not ref_ents:
        return {"med_entity_precision": 0.0, "med_entity_recall": 0.0, "med_entity_f1": 0.0}

    common = pred_ents & ref_ents
    precision = len(common) / len(pred_ents) if pred_ents else 0.0
    recall    = len(common) / len(ref_ents)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "med_entity_precision": precision,
        "med_entity_recall":    recall,
        "med_entity_f1":        f1,
    }


def retrieval_medical_coverage(
    retrieved_chunks: List[str],
    reference: str,
    scispacy_model: str = "en_core_sci_lg",
) -> float:
    """
    What fraction of ground-truth medical entities appear in
    the retrieved context?  (Retrieval-level medical recall)
    """
    ref_ents = set(_extract_entities_scispacy(reference, scispacy_model))
    if not ref_ents:
        return 0.0

    retrieved_text = " ".join(retrieved_chunks)
    retrieved_ents = set(_extract_entities_scispacy(retrieved_text, scispacy_model))

    covered = ref_ents & retrieved_ents
    return len(covered) / len(ref_ents)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PubMedQA Decision Accuracy
# ─────────────────────────────────────────────────────────────────────────────

_DECISION_KEYWORDS = {
    "yes":   {"yes", "positive", "confirmed", "true", "correct", "indeed", "affirmative"},
    "no":    {"no", "negative", "incorrect", "false", "not", "none", "neither"},
    "maybe": {"maybe", "possibly", "potentially", "unclear", "uncertain",
              "inconclusive", "mixed", "limited"},
}


def extract_decision(text: str) -> str:
    """
    Heuristically extract yes/no/maybe decision from generated answer text.
    Returns "yes", "no", "maybe", or "unknown".
    """
    text_lower = text.lower()

    # First try: explicit start of answer
    first_words = text_lower.split()[:5]
    for word in first_words:
        word = word.strip(".,;:()")
        for decision, keywords in _DECISION_KEYWORDS.items():
            if word in keywords:
                return decision

    # Second try: keyword frequency
    scores = {d: 0 for d in _DECISION_KEYWORDS}
    for decision, keywords in _DECISION_KEYWORDS.items():
        for kw in keywords:
            scores[decision] += text_lower.count(kw)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "unknown"


def pubmedqa_decision_accuracy(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Accuracy on yes/no/maybe classification.
    predictions and ground_truths are free-text strings.
    """
    correct = sum(
        extract_decision(pred) == gt.lower().strip()
        for pred, gt in zip(predictions, ground_truths)
    )
    return correct / len(predictions) if predictions else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregate medical scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_medical_metrics(
    prediction: str,
    reference: str,
    retrieved_chunks: List[str],
    final_decision_gt: Optional[str] = None,
    use_scispacy: bool = True,
    scispacy_model: str = "en_core_sci_lg",
) -> dict:
    """
    Compute all medical metrics for a single sample.

    Returns flat dict of metric_name → float.
    """
    results = {}

    # Medical entity coverage in retrieved context
    results["retrieval_medical_coverage"] = retrieval_medical_coverage(
        retrieved_chunks, reference, scispacy_model if use_scispacy else None
    ) if use_scispacy else 0.0

    # Terminology F1 between prediction and reference
    term_metrics = medical_terminology_f1(
        prediction, reference,
        scispacy_model if use_scispacy else "regex"
    )
    results.update(term_metrics)

    # Decision accuracy (if applicable)
    if final_decision_gt:
        predicted_decision = extract_decision(prediction)
        results["decision_match"] = float(
            predicted_decision == final_decision_gt.lower().strip()
        )
        results["predicted_decision"] = predicted_decision

    return results
