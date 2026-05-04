"""
report_generator.py
────────────────────────────────────────────────────────────────
Generates a polished HTML evaluation report from JSON results.
Also exports a summary CSV and a LaTeX-ready table for your paper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Metric display names  (raw key → pretty label)
METRIC_LABELS = {
    "context_precision_at_1":        "CP@1",
    "context_precision_at_3":        "CP@3",
    "context_precision_at_5":        "CP@5",
    "context_precision_at_10":       "CP@10",
    "context_recall_token":          "CR (token)",
    "context_recall_sentence":       "CR (sentence)",
    "legal_char_precision":          "Char Prec. (Legal)",
    "legal_char_recall":             "Char Recall (Legal)",
    "mrr":                           "MRR",
    "retrieval_medical_coverage":    "Med. Entity Coverage",
    "med_entity_precision":          "Med. Entity Prec.",
    "med_entity_recall":             "Med. Entity Recall",
    "med_entity_f1":                 "Med. Entity F1",
    "decision_match":                "Decision Acc. (PubMedQA)",
    "bioscore_f1":                   "BioScore F1",
    "ragas_context_precision":       "RAGAS CP (LLM)",
    "ragas_context_recall":          "RAGAS CR (LLM)",
    "ragas_faithfulness":            "Faithfulness (RAGAS)",
    "ragas_answer_relevancy":        "Answer Relevancy (RAGAS)",
}

VARIANT_LABELS = {
    "hybrid":           "🔀 Hybrid + Rerank (Ours)",
    "hybrid_norerank":  "🔀 Hybrid (no rerank)",
    "dense_only":       "🧠 Dense Only",
    "bm25_only":        "🔍 BM25 Only",
}

PRIMARY_METRICS = {
    "medical": ["context_precision_at_5", "context_recall_sentence",
                "med_entity_f1", "decision_match", "mrr"],
    "legal":   ["context_precision_at_5", "context_recall_sentence",
                "legal_char_precision", "legal_char_recall", "mrr"],
}


def load_results(results_dir: str) -> Dict[str, dict]:
    """Load all *_final.json files from results directory."""
    p = Path(results_dir)
    loaded = {}
    for f in sorted(p.glob("*_final.json")):
        with open(f) as fp:
            data = json.load(fp)
        key = f"{data['domain']}_{data['variant']}"
        loaded[key] = data
    return loaded


def build_summary_table(
    all_results: Dict[str, dict],
    domain: str,
    metrics: Optional[List[str]] = None,
) -> List[dict]:
    """
    Build a list of rows for a given domain's summary table.
    Each row: {variant, metric1, metric2, …}
    """
    if metrics is None:
        metrics = PRIMARY_METRICS.get(domain, list(METRIC_LABELS.keys()))

    rows = []
    variant_order = ["hybrid", "hybrid_norerank", "dense_only", "bm25_only"]
    for variant in variant_order:
        key = f"{domain}_{variant}"
        if key not in all_results:
            continue
        agg = all_results[key].get("aggregate", {})
        row = {"variant": VARIANT_LABELS.get(variant, variant)}
        for m in metrics:
            if m in agg:
                mean = agg[m]["mean"]
                std  = agg[m]["std"]
                row[METRIC_LABELS.get(m, m)] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[METRIC_LABELS.get(m, m)] = "—"
        rows.append(row)
    return rows


def results_to_latex_table(rows: List[dict], caption: str = "") -> str:
    """Generate LaTeX booktabs table from summary rows."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(r.get(h, "—")) for h in headers) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\label{{tab:rag_eval}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def generate_html_report(
    results_dir: str,
    output_path: str,
    experiment_name: str = "RAG Evaluation",
):
    """Generate the full HTML report."""
    all_results = load_results(results_dir)
    if not all_results:
        logger.error(f"No result files found in {results_dir}")
        return

    med_table = build_summary_table(all_results, "medical")
    leg_table = build_summary_table(all_results, "legal")

    med_latex = results_to_latex_table(
        med_table, caption="Medical RAG Evaluation Results (PubMedQA)"
    )
    leg_latex = results_to_latex_table(
        leg_table, caption="Legal RAG Evaluation Results (LegalBench-RAG)"
    )

    html = _render_html(
        experiment_name=experiment_name,
        med_table=med_table,
        leg_table=leg_table,
        med_latex=med_latex,
        leg_latex=leg_latex,
        all_results=all_results,
    )

    Path(output_path).write_text(html)
    logger.info(f"HTML report saved → {output_path}")

    # Save LaTeX tables separately
    latex_path = Path(output_path).with_suffix(".tex")
    latex_path.write_text(med_latex + "\n\n" + leg_latex)
    logger.info(f"LaTeX tables saved → {latex_path}")

    return output_path


def _table_html(rows: List[dict]) -> str:
    if not rows:
        return "<p>No results</p>"
    headers = list(rows[0].keys())
    th_row = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = ""
    for i, r in enumerate(rows):
        cls = "highlight" if i == 0 else ""
        tds = "".join(f"<td>{r.get(h, '—')}</td>" for h in headers)
        body_rows += f"<tr class='{cls}'>{tds}</tr>"
    return f"""
    <table>
      <thead><tr>{th_row}</tr></thead>
      <tbody>{body_rows}</tbody>
    </table>"""


def _render_html(
    experiment_name, med_table, leg_table, med_latex, leg_latex, all_results
) -> str:
    """Render the full HTML page."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    med_html = _table_html(med_table)
    leg_html = _table_html(leg_table)

    # Build chart data for Plotly
    def chart_data(domain, metrics_keys):
        variants = ["hybrid", "hybrid_norerank", "dense_only", "bm25_only"]
        traces = []
        for m_key in metrics_keys:
            y_vals, y_err, x_labels = [], [], []
            for v in variants:
                key = f"{domain}_{v}"
                if key in all_results:
                    agg = all_results[key].get("aggregate", {})
                    if m_key in agg:
                        y_vals.append(round(agg[m_key]["mean"], 4))
                        y_err.append(round(agg[m_key]["std"], 4))
                        x_labels.append(VARIANT_LABELS.get(v, v))
            if y_vals:
                label = METRIC_LABELS.get(m_key, m_key)
                traces.append({"name": label, "x": x_labels, "y": y_vals,
                                "error_y": y_err, "type": "bar"})
        return json.dumps(traces)

    med_metrics_keys = PRIMARY_METRICS["medical"]
    leg_metrics_keys = PRIMARY_METRICS["legal"]
    med_chart = chart_data("medical", med_metrics_keys)
    leg_chart = chart_data("legal", leg_metrics_keys)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{experiment_name} — RAG Evaluation Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {{
      --primary: #4f46e5; --bg: #f8f9fb; --card: #ffffff;
      --border: #e2e8f0; --text: #1e293b; --muted: #64748b;
      --green: #22c55e; --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg);
             color: var(--text); line-height: 1.6; }}
    .hero {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
              color: white; padding: 2.5rem 3rem; }}
    .hero h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }}
    .hero p  {{ opacity: .85; font-size: .95rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
    .card {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 12px; padding: 1.75rem; margin-bottom: 2rem;
              box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;
           color: var(--primary); }}
    h3 {{ font-size: 1.05rem; font-weight: 600; margin: 1.5rem 0 .75rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th {{ background: #f1f5f9; text-align: left; padding: .6rem .9rem;
           font-weight: 600; border-bottom: 2px solid var(--border); }}
    td {{ padding: .55rem .9rem; border-bottom: 1px solid var(--border); }}
    tr.highlight td {{ background: #eff6ff; font-weight: 600; }}
    tr:hover td {{ background: #f8faff; }}
    .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 99px;
               font-size: .75rem; font-weight: 600; background: #ede9fe;
               color: var(--primary); }}
    .latex-box {{ background: #1e1e2e; color: #cdd6f4; border-radius: 8px;
                   padding: 1.25rem; font-family: monospace; font-size: .8rem;
                   overflow-x: auto; white-space: pre; margin-top: .75rem; }}
    .chart-wrap {{ height: 420px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media(max-width:768px){{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<div class="hero">
  <h1>🔬 {experiment_name}</h1>
  <p>Advanced Hybrid RAG System · Evaluation Report · {timestamp}</p>
</div>

<div class="container">

  <!-- Medical -->
  <div class="card">
    <h2>🏥 Medical Domain — PubMedQA</h2>
    <p style="color:var(--muted);margin-bottom:1rem;font-size:.9rem">
      Hybrid BM25+Dense retrieval with BGE reranker evaluated on biomedical Q&A.
      Metrics: Context Precision@5, Context Recall (sentence), Medical Entity F1,
      PubMedQA Decision Accuracy, MRR.
    </p>
    {med_html}
    <div class="chart-wrap" id="medChart" style="margin-top:1.5rem"></div>
  </div>

  <!-- Legal -->
  <div class="card">
    <h2>⚖️ Legal Domain — LegalBench-RAG</h2>
    <p style="color:var(--muted);margin-bottom:1rem;font-size:.9rem">
      Character-level precision retrieval on legal corpora (79M chars, expert-annotated).
      Metrics: Context Precision@5, Context Recall, Char Precision/Recall, MRR.
    </p>
    {leg_html}
    <div class="chart-wrap" id="legChart" style="margin-top:1.5rem"></div>
  </div>

  <!-- LaTeX tables -->
  <div class="card">
    <h2>📄 Paper-Ready LaTeX Tables</h2>
    <h3>Medical (PubMedQA)</h3>
    <div class="latex-box">{med_latex}</div>
    <h3>Legal (LegalBench-RAG)</h3>
    <div class="latex-box">{leg_latex}</div>
  </div>

</div>

<script>
const layout = {{
  barmode: 'group',
  plot_bgcolor: '#fff',
  paper_bgcolor: '#fff',
  font: {{ family: 'Inter, sans-serif', size: 12 }},
  yaxis: {{ range: [0, 1], gridcolor: '#e2e8f0', title: 'Score' }},
  xaxis: {{ gridcolor: '#e2e8f0' }},
  legend: {{ orientation: 'h', y: -0.2 }},
  margin: {{ t: 30, b: 80 }},
}};
const cfg = {{ responsive: true }};

Plotly.newPlot('medChart', {med_chart}, {{...layout, title: 'Medical Metrics by Variant'}}, cfg);
Plotly.newPlot('legChart', {leg_chart}, {{...layout, title: 'Legal Metrics by Variant'}}, cfg);
</script>
</body>
</html>"""
