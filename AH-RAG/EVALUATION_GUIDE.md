# RAG Evaluation Guide
## Advanced Hybrid RAG System · Medical & Legal Domains
### For Paper Submission (deadline-ready)

---

## Directory Structure

```
rag_evaluation/
├── eval_config.yaml          ← All settings live here
├── requirements_eval.txt     ← pip install this
├── local_rag_system.py       ← Your RAG with local models (NVIDIA → BGE/Ollama)
├── eval_pipeline.py          ← Core evaluation loop
├── run_evaluation.py         ← CLI entry point
├── report_generator.py       ← HTML report + LaTeX tables
├── datasets/
│   ├── pubmedqa_loader.py    ← PubMedQA (HuggingFace)
│   └── legalbench_loader.py  ← LegalBench-RAG (GitHub auto-clone)
└── metrics/
    ├── context_metrics.py    ← Context Precision & Recall (token + RAGAS)
    └── medical_metrics.py    ← BioScore, Entity F1, Decision Accuracy
```

---

## Step 1 · Environment Setup

```bash
# Create isolated environment
conda create -n rag_eval python=3.10 -y
conda activate rag_eval

# Install all dependencies
pip install -r requirements_eval.txt

# Medical NLP model (large — needs internet once)
python -m spacy download en_core_sci_lg
# If en_core_sci_lg fails, use the smaller version:
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Verify NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Step 2 · Install & Start Ollama (Local LLM)

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (keep this terminal open)
ollama serve

# Pull your LLM — in a new terminal:
ollama pull llama3:70b        # Best quality (~40GB VRAM)
# OR if VRAM is tight:
ollama pull llama3:8b         # Faster (~8GB VRAM)
ollama pull mistral:7b        # Good alternative

# Test it's working
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3:70b","prompt":"Say hello","stream":false}'
```

> **vLLM alternative** (if you prefer):
> ```bash
> pip install vllm
> python -m vllm.entrypoints.openai.api_server \
>   --model meta-llama/Meta-Llama-3-70B-Instruct \
>   --tensor-parallel-size 2    # adjust for your GPUs
> ```
> Then set `llm_backend: "vllm"` in `eval_config.yaml`.

---

## Step 3 · Configure eval_config.yaml

Open `eval_config.yaml` and adjust:

```yaml
models:
  llm_backend: "ollama"         # Change to "vllm" if using vLLM
  ollama:
    model: "llama3:70b"         # Match what you pulled
    
  embedding:
    device: "cuda"              # or "cpu"

datasets:
  pubmedqa:
    num_samples: 200            # 200 for paper; 50 for quick test

  legalbench_rag:
    use_mini: true              # false for full benchmark
    num_samples: 150
```

---

## Step 4 · Run the Evaluation

### Option A: Full evaluation (recommended for paper)
```bash
cd rag_evaluation
python run_evaluation.py
```
This runs all 4 variants × 2 domains = 8 evaluation runs.
Takes ~2–4 hours for 200 medical + 150 legal samples.

### Option B: Fast sanity check first (5–10 min)
```bash
python run_evaluation.py --fast
```
50 samples, hybrid only, no RAGAS — good to verify everything works.

### Option C: Domain-specific
```bash
# Medical only
python run_evaluation.py --domain medical --variants hybrid dense_only bm25_only

# Legal only  
python run_evaluation.py --domain legal --ragas
```

### Option D: With RAGAS LLM-judged metrics
```bash
python run_evaluation.py --ragas
```
Adds RAGAS Context Precision/Recall/Faithfulness (LLM-judged, slower).

---

## Step 5 · View Results

```bash
# Regenerate HTML report from saved results (fast)
python run_evaluation.py --report-only

# Open in browser
xdg-open reports/hybrid_rag_v1_report.html    # Linux
open reports/hybrid_rag_v1_report.html        # Mac
```

The report includes:
- Comparison table: Hybrid vs Dense-only vs BM25-only vs Hybrid-no-rerank
- Bar charts per domain
- **Copy-paste LaTeX tables** for your paper
- BioScore (BioBERT BERTScore) for medical factuality

---

## Step 6 · Raw Results

Individual JSON result files saved in `results/`:
```
results/
├── medical_hybrid_final.json
├── medical_dense_only_final.json
├── medical_bm25_only_final.json
├── medical_hybrid_norerank_final.json
├── legal_hybrid_final.json
├── legal_dense_only_final.json
├── ...
```

Each file structure:
```json
{
  "domain": "medical",
  "variant": "hybrid",
  "aggregate": {
    "context_precision_at_5": {"mean": 0.82, "std": 0.12, "n": 200},
    "context_recall_sentence": {"mean": 0.74, "std": 0.15, "n": 200},
    "mrr": {"mean": 0.71, "std": 0.18, "n": 200},
    ...
  },
  "samples": [...]
}
```

---

## Metrics Reference

| Metric | Domain | Description |
|---|---|---|
| **Context Precision@K** | Both | Fraction of top-K retrieved chunks that are relevant. Weighted (RAGAS-style): higher credit for relevant chunks ranked first. |
| **Context Recall (sentence)** | Both | Fraction of ground-truth sentences covered by retrieved chunks. |
| **Context Recall (token)** | Both | Fraction of ground-truth tokens present in retrieved context. |
| **MRR** | Both | Mean Reciprocal Rank — reciprocal of first relevant chunk's rank. |
| **Char Precision/Recall** | Legal | LegalBench-RAG's native metric — character-level overlap with ground-truth legal span. |
| **Medical Entity F1** | Medical | Precision/Recall/F1 on UMLS entities (via scispaCy) between predicted and reference answer. |
| **Decision Accuracy** | Medical | Yes/No/Maybe classification accuracy on PubMedQA final_decision. |
| **BioScore F1** | Medical | BERTScore using BioBERT — measures semantic factual alignment. |
| **RAGAS CP/CR** | Both (optional) | LLM-judged Context Precision/Recall — more nuanced but needs Ollama. |

---

## Ablation Variants

The pipeline automatically tests:

| Variant | Description |
|---|---|
| `hybrid` | BM25 + Dense + BGE Reranker **(your system)** |
| `hybrid_norerank` | BM25 + Dense fusion, no reranker |
| `dense_only` | BGE embeddings + ChromaDB only |
| `bm25_only` | BM25 keyword search only |

This demonstrates the contribution of each component for your paper.

---

## Troubleshooting

**`ollama: command not found`**
→ `export PATH=$PATH:~/.ollama/bin` or restart shell.

**CUDA out of memory**
→ Use `llama3:8b` instead of 70b, or set `device: "cpu"` in config.

**scispaCy model not found**
→ Set `use_scispacy_ner: false` in `eval_config.yaml` — falls back to regex.

**LegalBench clone fails**
→ `pip install gitpython` or manually clone:
   `git clone --depth=1 https://github.com/zeroentropy-cc/legalbenchrag ./legalbenchrag_data`

**RAGAS import error**
→ `pip install ragas==0.1.9` (version pinned for LlamaIndex compatibility).

**`nest_asyncio` / event loop errors**
→ Add `import nest_asyncio; nest_asyncio.apply()` at the top of your script.

---

## Paper Writing Tips

The LaTeX tables are auto-generated in `reports/hybrid_rag_v1_report.tex`.

For the ablation table in your paper, the key comparison rows are:
- **Hybrid + Rerank** (your contribution)
- **Dense Only** (NVIDIA embedding baseline)
- **BM25 Only** (sparse baseline)

Expected hybrid advantages (from literature):
- Medical: BM25 captures rare drug names / gene symbols that semantic search misses.
- Legal: Exact legal citations (e.g. "§ 12(b)(6)") retrieved by BM25; BGE handles
  semantic paraphrase of legal concepts.

---

*Generated by the RAG Evaluation Suite · Local-first, GPU-accelerated.*
