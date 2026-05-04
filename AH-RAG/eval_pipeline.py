"""
eval_pipeline.py
────────────────────────────────────────────────────────────────
Orchestrates the full evaluation loop:

  For each domain (medical / legal):
    1. Load dataset samples
    2. For each sample:
       a. Ingest corpus (abstract or document) into local RAG
       b. Run retrieval  → retrieved_chunks
       c. Run generation → predicted_answer
       d. Compute metrics (CP, CR, domain-specific)
    3. Aggregate + save results
    4. (Optionally) run RAGAS LLM-judged metrics in batch

Ablation variants run automatically:
  - dense_only   (no BM25)
  - bm25_only    (no dense)
  - hybrid       (our system, with reranker)
  - hybrid_norerank (hybrid without reranker)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from local_rag_system import LocalRAGSystem
from context_metrics import compute_retrieval_metrics, compute_ragas_metrics
from medical_metrics import compute_medical_metrics, compute_bioscores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    domain: str
    variant: str                        # "hybrid" | "dense_only" | "bm25_only" | "hybrid_norerank"
    question: str
    ground_truth: str
    retrieved_chunks: List[str]
    generated_answer: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalResults:
    domain: str
    variant: str
    samples: List[SampleResult] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        """Compute mean ± std for each metric across all samples."""
        if not self.samples:
            return {}
        all_metrics: Dict[str, List[float]] = {}
        for s in self.samples:
            for k, v in s.metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_metrics.setdefault(k, []).append(float(v))
        return {
            k: float(np.mean(v)) for k, v in all_metrics.items()
        }

    def aggregate_with_std(self) -> Dict[str, Dict[str, float]]:
        """Returns {metric: {mean: x, std: y}}."""
        if not self.samples:
            return {}
        all_metrics: Dict[str, List[float]] = {}
        for s in self.samples:
            for k, v in s.metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_metrics.setdefault(k, []).append(float(v))
        return {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in all_metrics.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Runs the full evaluation pipeline for one or more domains.
    """

    VARIANTS = ["hybrid", "hybrid_norerank", "dense_only", "bm25_only"]

    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config["output"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._rag: Optional[LocalRAGSystem] = None

    # ──────────────────────────────────────────────────────────────────────
    # Model initialisation (shared across all evaluations)
    # ──────────────────────────────────────────────────────────────────────
    def init_models(self) -> bool:
        logger.info("Initialising local models …")
        self._rag = LocalRAGSystem(self.config)
        ok = self._rag.setup_models()
        if not ok:
            logger.error("Model initialisation failed")
        return ok

    # ──────────────────────────────────────────────────────────────────────
    # Medical evaluation (PubMedQA)
    # ──────────────────────────────────────────────────────────────────────
    def run_medical_eval(
        self,
        variants: Optional[List[str]] = None,
        run_ragas: bool = False,
    ) -> Dict[str, EvalResults]:
        from pubmedqa_loader import load_pubmedqa

        variants = variants or self.VARIANTS
        cfg = self.config["datasets"]["pubmedqa"]

        logger.info("=" * 60)
        logger.info("MEDICAL EVALUATION  (PubMedQA)")
        logger.info("=" * 60)

        samples = load_pubmedqa(
            num_samples=cfg["num_samples"],
        )

        all_results: Dict[str, EvalResults] = {}
        for variant in variants:
            logger.info(f"\n── Variant: {variant} ──")
            eval_res = EvalResults(domain="medical", variant=variant)

            for i, sample in enumerate(tqdm(samples, desc=f"medical/{variant}")):
                sr = self._eval_single_medical(sample, variant, idx=i)
                eval_res.samples.append(sr)

                # Save periodically
                if (i + 1) % 50 == 0:
                    self._save_partial(eval_res)

            all_results[variant] = eval_res
            self._save_domain_results(eval_res)

        # Optional RAGAS batch pass (hybrid variant only)
        if run_ragas and self.config["ragas"]["use_llm_judge"]:
            self._run_ragas_pass(all_results.get("hybrid"), domain="medical")

        return all_results

    # ──────────────────────────────────────────────────────────────────────
    # Legal evaluation (LegalBench-RAG)
    # ──────────────────────────────────────────────────────────────────────
    def run_legal_eval(
        self,
        variants: Optional[List[str]] = None,
        run_ragas: bool = False,
    ) -> Dict[str, EvalResults]:
        from legalbench_loader import load_legalbench_rag

        variants = variants or self.VARIANTS
        cfg = self.config["datasets"]["legalbench_rag"]

        logger.info("=" * 60)
        logger.info("LEGAL EVALUATION  (LegalBench-RAG)")
        logger.info("=" * 60)

        samples = load_legalbench_rag(
            cuad_json_path=cfg.get("cuad_json_path", "/tmp/CUAD_v1.json"),
            num_samples=cfg["num_samples"],
        )

        all_results: Dict[str, EvalResults] = {}
        for variant in variants:
            logger.info(f"\n── Variant: {variant} ──")
            eval_res = EvalResults(domain="legal", variant=variant)

            for i, sample in enumerate(tqdm(samples, desc=f"legal/{variant}")):
                sr = self._eval_single_legal(sample, variant, idx=i)
                eval_res.samples.append(sr)

                if (i + 1) % 50 == 0:
                    self._save_partial(eval_res)

            all_results[variant] = eval_res
            self._save_domain_results(eval_res)

        if run_ragas and self.config["ragas"]["use_llm_judge"]:
            self._run_ragas_pass(all_results.get("hybrid"), domain="legal")

        return all_results

    # ──────────────────────────────────────────────────────────────────────
    # Single-sample evaluation helpers
    # ──────────────────────────────────────────────────────────────────────
    def _eval_single_medical(self, sample, variant: str, idx: int) -> SampleResult:
        """Evaluate one PubMedQA sample."""
        from pubmedqa_loader import PubMedQASample
        t0 = time.perf_counter()

        sr = SampleResult(
            sample_id=f"medical_{idx}",
            domain="medical",
            variant=variant,
            question=sample.question,
            ground_truth=sample.long_answer,
            retrieved_chunks=[],
            generated_answer="",
        )

        try:
            # Ingest this sample's abstract as the corpus
            ok, _, _ = self._rag.ingest_documents_from_texts(
                [sample.corpus_text],
                metadata_list=[sample.metadata],
                collection_name=f"eval_medical_{idx}",
            )
            if not ok:
                sr.error = "ingestion failed"
                return sr

            ok = self._rag.setup_retrieval()
            if not ok:
                sr.error = "retrieval setup failed"
                return sr

            # Retrieve
            sr.retrieved_chunks = self._retrieve_by_variant(
                sample.question, variant, domain="medical"
            )

            # Generate answer
            sr.generated_answer = self._rag.query(sample.question)

            # Compute retrieval metrics
            sr.metrics.update(compute_retrieval_metrics(
                sr.retrieved_chunks,
                ground_truth=sample.ground_truth_text,
                domain="medical",
            ))

            # Medical domain metrics
            med_metrics = compute_medical_metrics(
                prediction=sr.generated_answer,
                reference=sample.long_answer,
                retrieved_chunks=sr.retrieved_chunks,
                final_decision_gt=sample.final_decision,
                use_scispacy=self.config["domain_metrics"]["medical"]["use_scispacy_ner"],
                scispacy_model=self.config["domain_metrics"]["medical"]["scispacy_model"],
            )
            sr.metrics.update(med_metrics)

        except Exception as exc:
            logger.warning(f"Sample {idx} failed: {exc}")
            sr.error = str(exc)

        sr.latency_s = time.perf_counter() - t0
        return sr

    def _eval_single_legal(self, sample, variant: str, idx: int) -> SampleResult:
        """Evaluate one LegalBench-RAG sample."""
        t0 = time.perf_counter()

        sr = SampleResult(
            sample_id=f"legal_{idx}",
            domain="legal",
            variant=variant,
            question=sample.query,
            ground_truth=sample.ground_truth_context,
            retrieved_chunks=[],
            generated_answer="",
        )

        try:
            if not sample.document_text.strip():
                sr.error = "empty document"
                return sr

            ok, _, _ = self._rag.ingest_documents_from_texts(
                [sample.document_text],
                metadata_list=[sample.metadata],
                collection_name=f"eval_legal_{idx}",
            )
            if not ok:
                sr.error = "ingestion failed"
                return sr

            ok = self._rag.setup_retrieval()
            if not ok:
                sr.error = "retrieval setup failed"
                return sr

            sr.retrieved_chunks = self._retrieve_by_variant(
                sample.query, variant, domain="legal"
            )
            sr.generated_answer = self._rag.query(sample.query)

            # Retrieval metrics (standard + legal char-level)
            sr.metrics.update(compute_retrieval_metrics(
                sr.retrieved_chunks,
                ground_truth=sample.relevant_span,
                domain="legal",
            ))

        except Exception as exc:
            logger.warning(f"Sample {idx} failed: {exc}")
            sr.error = str(exc)

        sr.latency_s = time.perf_counter() - t0
        return sr

    def _retrieve_by_variant(
        self, query: str, variant: str, domain: str = "general"
    ) -> List[str]:
        """Route to the correct retrieval variant, applying query reformulation by domain."""
        if variant == "hybrid":
            return self._rag.retrieve_texts(query, domain=domain)
        elif variant == "dense_only":
            return self._rag.retrieve_dense_only(query, domain=domain)
        elif variant == "bm25_only":
            return self._rag.retrieve_bm25_only(query, domain=domain)
        elif variant == "hybrid_norerank":
            saved = self._rag.reranker
            self._rag.reranker = None
            self._rag.setup_retrieval()
            result = self._rag.retrieve_texts(query, domain=domain)
            self._rag.reranker = saved
            self._rag.setup_retrieval()
            return result
        else:
            raise ValueError(f"Unknown variant: {variant}")

    # ──────────────────────────────────────────────────────────────────────
    # RAGAS batch pass
    # ──────────────────────────────────────────────────────────────────────
    def _run_ragas_pass(self, eval_res: Optional[EvalResults], domain: str):
        if eval_res is None or not eval_res.samples:
            return
        logger.info(f"Running RAGAS LLM-judged metrics for {domain}/hybrid …")

        valid = [s for s in eval_res.samples if not s.error and s.generated_answer]
        dataset = {
            "question":     [s.question for s in valid],
            "answer":       [s.generated_answer for s in valid],
            "contexts":     [s.retrieved_chunks for s in valid],
            "ground_truth": [s.ground_truth for s in valid],
        }
        try:
            ragas_scores = compute_ragas_metrics(
                dataset_dict=dataset,
                llm=self._rag.llm,
                embeddings=self._rag.embed_model,
                metrics_to_run=self.config["ragas"]["metrics"],
            )
            logger.info(f"RAGAS scores ({domain}): {ragas_scores}")

            # Attach back to individual samples
            for k, mean_val in ragas_scores.items():
                for s in valid:
                    s.metrics[f"ragas_{k}"] = mean_val   # broadcast mean

            self._save_domain_results(eval_res)

        except Exception as exc:
            logger.warning(f"RAGAS pass failed: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    # I/O
    # ──────────────────────────────────────────────────────────────────────
    def _save_partial(self, eval_res: EvalResults):
        path = self.results_dir / f"{eval_res.domain}_{eval_res.variant}_partial.json"
        self._write_json(path, eval_res)

    def _save_domain_results(self, eval_res: EvalResults):
        path = self.results_dir / f"{eval_res.domain}_{eval_res.variant}_final.json"
        self._write_json(path, eval_res)
        logger.info(f"Saved results → {path}")
        agg = eval_res.aggregate()
        logger.info(f"Aggregate ({eval_res.domain}/{eval_res.variant}): {agg}")

    @staticmethod
    def _write_json(path: Path, eval_res: EvalResults):
        data = {
            "domain":  eval_res.domain,
            "variant": eval_res.variant,
            "aggregate": eval_res.aggregate_with_std(),
            "samples": [asdict(s) for s in eval_res.samples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
