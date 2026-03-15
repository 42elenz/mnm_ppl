# Marburg-Mannheim: Perplexity Transdiagnostic Analyses

<p align="center">
  <img src="assets/mnm.png" alt="MNM" width="70%"/>
</p>

Normative modelling pipeline that uses **LLM perplexity** as a proxy for formal thought disorder in TAT (Thematic Apperception Test) transcripts. Sentences from psychiatric patients (SSD, MDD, BD) and healthy controls are scored by how "surprising" they are to a language model under different prompting conditions.

---

## Project Structure

```
mnm_ppl/
├── calculate_perplexities.py        # Main entry point (Steps 1–2)
├── perplexity_analysis.ipynb        # Analysis notebook (Steps 3–9)
├── pyproject.toml                   # Dependencies (managed by uv)
├── services/
│   ├── preprocessing_service.py     # Orchestrates transcript parsing
│   ├── perplexity_service.py        # Orchestrates vLLM perplexity computation
│   ├── analysis_service.py          # Loads data, aggregates to person level
│   └── plotting_service.py          # Generates all plots per analysis setting
├── functions/
│   ├── preprocessing/
│   │   └── transkript_parser.py     # Parses raw TAT .txt files → utterance CSV
│   ├── ppl/
│   │   ├── compute_perplexities.py  # Sentence splitting, prompt setup, PPL call
│   │   └── helper_perplexity.py     # vLLM inference, token-level perplexity
│   ├── analysis/
│   │   └── statistics.py            # Aggregation, group tests, correlations, ROC
│   └── plotting/
│       └── plots.py                 # All plot functions (11 plot types)
└── data/
    └── 00_raw_data/
        └── Demografics_Speech_FOR2107.csv
```

---

## Setup

Requires **Python ≥ 3.12** and [**uv**](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/42elenz/mnm_ppl.git
cd mnm_ppl
uv sync
```

This creates a `.venv/` and installs all dependencies from `pyproject.toml`.

### GPU Requirements

Perplexity computation uses [vLLM](https://docs.vllm.ai/) with tensor parallelism. The number of GPUs is detected automatically (supports 1, 2, 4, or 8). For `Qwen2.5-32B-Instruct`, at least **2× A100 80 GB** (or equivalent) are recommended. Smaller models like `Qwen2.5-7B-Instruct` run on a single GPU.

---

## Usage

The pipeline has two phases:

### Step 1 + 2: Preprocessing & Perplexity Computation

```bash
uv run python calculate_perplexities.py
```

This runs:

1. **Preprocessing** — Parses raw TAT `.txt` files into a sentence-level CSV. Each file contains transcribed speech from one participant across four TAT picture descriptions (Bild 1, 2, 4, 6). Utterances are split into sentences at `.!?` boundaries.

2. **Perplexity computation** — Loads a vLLM model and computes token-level log-probabilities for each sentence under three **prompt conditions**:
   - `healthy` — System prompt describes a mentally healthy person
   - `sick` — System prompt describes a person with a mental illness
   - `neutral` — Minimal system prompt with no clinical framing

   For each sentence, two perplexity scores are computed:
   - `perplexity_full` — over all tokens (prompt + sentence)
   - `perplexity_completion` — over sentence tokens only ← **primary metric**

   Results are cached to CSV; re-running skips already computed conditions (`overwrite=False`).

**Configuration** (edit `calculate_perplexities.py`):
```python
INPUT_TEXTS_DIR   = "path/to/raw/TAT/text/files"
UTTERANCE_CSV     = "path/to/output/utterances.csv"
PERPLEXITY_OUTPUT = "data/02_perplexities"
MODEL_NAME        = "Qwen/Qwen2.5-7B-Instruct"   # or Qwen2.5-32B-Instruct
```

### Steps 3–9: Analysis

Open and run the Jupyter notebook:

```bash
uv run jupyter lab perplexity_analysis.ipynb
```

The notebook:

1. **Loads** sentence-level perplexities + demographics/clinical metadata
2. **Aggregates** to person level (mean, median, std, p90, p75, CV, proportion of high-PPL sentences)
3. **Prepares three analysis settings**:
   - `with_outliers` — raw perplexity values
   - `without_outliers` — winsorized (5%) + log-transformed
   - `matched` — 1:1 optimal matching on age, sex, education
4. **Group comparisons** — t-test, Mann-Whitney U, Cohen's d (HC vs patients; pairwise by diagnosis)
5. **Clinical correlations** — Spearman r with GAF, TLI subscales (Impoverished, Disorganization, Non-specific)
6. **Alternative metrics comparison** — Compares all aggregation strategies (mean, median, std, p90, p75, CV, high-PPL proportion)
7. **Steiger's Z** — Tests whether TLI Impoverished and TLI Disorganization correlate differently with perplexity
8. **ROC analysis** — AUC for predicting clinical impairment (GAF < 60, TLI > 0); logistic regression with 5-fold CV
9. **Confounder analysis** — Nested OLS regressions controlling for word count and education; residualized perplexity plots

All figures are saved to `figures/<setting_name>/`.

---

## Key Packages

| Package | Purpose |
|---------|---------|
| **[vllm](https://docs.vllm.ai/)** | High-throughput LLM inference with tensor parallelism. Used to compute token-level log-probabilities for perplexity scores. |
| **[pandas](https://pandas.pydata.org/)** | Data manipulation throughout the pipeline (transcript parsing, aggregation, merging with metadata). |
| **[numpy](https://numpy.org/)** | Numerical operations (log-prob → perplexity conversion, aggregation metrics). |
| **[scipy](https://scipy.org/)** | Statistical tests (t-test, Mann-Whitney U, Kruskal-Wallis, Spearman correlation, winsorization). |
| **[statsmodels](https://www.statsmodels.org/)** | OLS regression for confounder analysis and nested models. |
| **[scikit-learn](https://scikit-learn.org/)** | Logistic regression with cross-validated AUC; StandardScaler for matching. |
| **[matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/)** | All visualizations (boxplots, scatter plots, ROC curves, heatmaps, violin plots). |
| **[tqdm](https://tqdm.github.io/)** | Progress bars during perplexity computation. |

---

## Adding Dependencies

```bash
uv add <package>
```

This updates both `pyproject.toml` and `uv.lock`.