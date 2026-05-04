#!/usr/bin/env python3
"""
run_evaluation.py
────────────────────────────────────────────────────────────────
CLI entry point for the RAG evaluation suite.

Quick start
───────────
  # Full eval (medical + legal, all variants, no RAGAS LLM judge)
  python run_evaluation.py

  # Medical only, hybrid variant, with RAGAS
  python run_evaluation.py --domain medical --variants hybrid --ragas

  # Just generate a report from existing results
  python run_evaluation.py --report-only

  # Use custom config
  python run_evaluation.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich import box

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "eval_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_banner():
    console.rule("[bold purple]Advanced Hybrid RAG — Evaluation Suite[/]")
    console.print(
        "\n  [bold]Domains[/]: Medical (PubMedQA) · Legal (LegalBench-RAG)"
        "\n  [bold]Metrics[/]: Context Precision · Context Recall · "
        "Medical Entity F1 · Decision Accuracy · Char Precision/Recall · MRR\n"
    )


def print_summary(results_dir: str):
    """Print a rich summary table from saved results."""
    p = Path(results_dir)
    files = sorted(p.glob("*_final.json"))
    if not files:
        console.print("[yellow]No result files found yet[/]")
        return

    table = Table(
        title="Evaluation Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Domain",   style="bold")
    table.add_column("Variant",  style="dim")
    table.add_column("CP@5",     justify="right")
    table.add_column("CR (sent)", justify="right")
    table.add_column("MRR",      justify="right")
    table.add_column("Samples",  justify="right")

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        domain  = data.get("domain", "?")
        variant = data.get("variant", "?")
        agg     = data.get("aggregate", {})
        n       = len(data.get("samples", []))

        cp5 = agg.get("context_precision_at_5", {}).get("mean", 0)
        cr  = agg.get("context_recall_sentence", {}).get("mean", 0)
        mrr = agg.get("mrr", {}).get("mean", 0)

        style = "bold green" if variant == "hybrid" else ""
        table.add_row(
            domain, variant,
            f"[{style}]{cp5:.3f}[/]",
            f"[{style}]{cr:.3f}[/]",
            f"[{style}]{mrr:.3f}[/]",
            str(n),
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", default="eval_config.yaml",
        help="Path to eval_config.yaml (default: eval_config.yaml)"
    )
    parser.add_argument(
        "--domain", choices=["medical", "legal", "both"], default="both",
        help="Which domain to evaluate"
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=["hybrid", "hybrid_norerank", "dense_only", "bm25_only"],
        default=None,
        help="Which retrieval variants to test (default: all)"
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Run RAGAS LLM-judged metrics (requires Ollama running)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip evaluation; just regenerate HTML report from existing results"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override number of samples (overrides config)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 50 samples, hybrid variant only, no RAGAS"
    )
    args = parser.parse_args()

    print_banner()

    # ── Load config ──────────────────────────────────────────────────────
    config = load_config(args.config)

    if args.fast:
        config["datasets"]["pubmedqa"]["num_samples"]       = 50
        config["datasets"]["legalbench_rag"]["num_samples"] = 50
        args.variants = ["hybrid"]
        args.ragas    = False
        console.print("[yellow]⚡ FAST MODE: 50 samples, hybrid only[/]\n")

    if args.samples:
        config["datasets"]["pubmedqa"]["num_samples"]       = args.samples
        config["datasets"]["legalbench_rag"]["num_samples"] = args.samples

    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    exp_name    = config["output"]["experiment_name"]

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # ── Report only ───────────────────────────────────────────────────────
    if args.report_only:
        console.print("[bold]Generating report from existing results …[/]")
        from report_generator import generate_html_report
        out = f"{reports_dir}/{exp_name}_report.html"
        generate_html_report(results_dir, out, exp_name)
        console.print(f"[green]Report: {out}[/]")
        return

    # ── Run evaluation ────────────────────────────────────────────────────
    from eval_pipeline import RAGEvaluator

    evaluator = RAGEvaluator(config)

    console.print("[bold cyan]Initialising local models …[/]")
    if not evaluator.init_models():
        console.print("[red]Model init failed. Check vLLM server or model path:[/]")
        console.print("  [dim]python -m vllm.entrypoints.openai.api_server \\"
                      "--model /home/paritosh/casal/models/Meta-Llama-3-8B "
                      "--served-model-name meta-llama/Meta-Llama-3-8B --port 8000[/]")
        sys.exit(1)
    console.print("[green]Models ready ✓[/]\n")

    t_start = time.perf_counter()

    all_eval_results = {}

    if args.domain in ("medical", "both"):
        console.rule("[bold blue]Medical Evaluation[/]")
        med_results = evaluator.run_medical_eval(
            variants=args.variants, run_ragas=args.ragas
        )
        all_eval_results["medical"] = med_results

    if args.domain in ("legal", "both"):
        console.rule("[bold blue]Legal Evaluation[/]")
        leg_results = evaluator.run_legal_eval(
            variants=args.variants, run_ragas=args.ragas
        )
        all_eval_results["legal"] = leg_results

    elapsed = time.perf_counter() - t_start
    console.print(f"\n[green]Evaluation complete in {elapsed/60:.1f} min[/]")

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(results_dir)

    # ── Generate report ────────────────────────────────────────────────────
    if config["output"]["generate_html_report"]:
        from report_generator import generate_html_report
        out = f"{reports_dir}/{exp_name}_report.html"
        generate_html_report(results_dir, out, exp_name)
        console.print(f"\n[green]📊 Report saved → {out}[/]")

    # ── BioScore post-pass (requires GPU, run separately if slow) ─────────
    _maybe_run_bioscores(all_eval_results, results_dir, config)


def _maybe_run_bioscores(all_eval_results, results_dir, config):
    """Run BERTScore (BioScore) post-pass for medical if answers are available."""
    if "medical" not in all_eval_results:
        return
    hybrid = all_eval_results["medical"].get("hybrid")
    if not hybrid:
        return

    valid = [s for s in hybrid.samples if s.generated_answer and s.ground_truth]
    if not valid:
        return

    console.print("[bold cyan]Running BioScore (BioBERT BERTScore) …[/]")
    try:
        from medical_metrics import compute_bioscores
        _device = config["models"]["embedding"].get("device", "cpu")
        bios = compute_bioscores(
            predictions=[s.generated_answer for s in valid],
            references=[s.ground_truth for s in valid],
            model_name=config["domain_metrics"]["medical"]["bertscore_model"],
            device=_device,
        )
        mean_f1 = sum(bios["bioscore_f1"]) / len(bios["bioscore_f1"])
        console.print(f"  BioScore F1 (hybrid) = [green]{mean_f1:.4f}[/]")

        # Attach to samples
        for i, s in enumerate(valid):
            s.metrics["bioscore_f1"] = bios["bioscore_f1"][i]

        # Re-save
        from eval_pipeline import RAGEvaluator
        RAGEvaluator._write_json(
            Path(results_dir) / "medical_hybrid_final.json", hybrid
        )
    except Exception as e:
        logger.warning(f"BioScore failed: {e}")


if __name__ == "__main__":
    main()
