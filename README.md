# Effective Context in Transformers: An Analysis of Fragmentation and Tokenization

Reproducibility code for the paper.

## Setup

The conda environment is named `bane` (legacy shorthand from the previous working title).

### Create the environment (GPU machine, CUDA 12.8+)

```bash
conda env create -f environment.yml
conda activate bane
```

`environment.yml` pulls CUDA-enabled PyTorch wheels via the `cu128` index,
which is compatible with CUDA 12.8 through 13.x.

### Manual install (if `conda env create` is unavailable)

```bash
conda create -n bane python=3.11 -y
conda activate bane
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### CPU-only machine

Remove the `--extra-index-url` line from `environment.yml` and replace
the pinned `torch==2.11.0` / `torchvision` / `torchaudio` lines in
`requirements.txt` with their plain PyPI versions, then run
`conda env create -f environment.yml`.

### Verify

```bash
conda run -n bane python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Reproducing Paper Results

Each experiment is self-contained: run the training script, then the plot script.
GPU is required for the transformer experiments; the n-gram and tokenizer
span experiments run on CPU.

### Section 3 — Fragmentation in Transformer

Trains a symbol-level and a fragment-level transformer on a Markov source and
compares their cross-entropy curves to the theoretical bound.

```bash
# Train (GPU; 3 seeds, ~1–2 h on a single GPU)
conda run -n bane python scripts/run_fragmentation_transformer.py \
    configs/fragmentation_transformer/final.yaml

# Regenerate figures from saved logs (no retraining needed)
conda run -n bane python scripts/plot_fragmentation_transformer_final.py
```

Figures are written to `outputs/fragmentation_transformer/figures/`:
- `main/fragmentation_transformer_final.{pdf,png}` — main paper figure
- `appendix/fragmentation_transformer_final_grad.{pdf,png}` — gradient-norm appendix figure

### Section 3 — Fragmentation Terms Decomposition

Measures the excess loss of an n-gram model that operates on fragments
rather than source symbols, across a grid of Markov orders and alphabet sizes.

```bash
# Run base configuration (K = k, exact context match)
conda run -n bane python scripts/run_fragmentation_ngram.py

# Run K = k+1 variant (one extra token of context)
conda run -n bane python scripts/run_fragmentation_ngram.py \
    configs/fragmentation_ngram/Kp1.yaml

# Sweep over Dirichlet alpha and context offset K - k
conda run -n bane python scripts/sweep_fragmentation_ngram.py
```

Artifacts are written to `outputs/fragmentation_ngram/`
(figures and raw CSVs/JSONs).

### Section 4 — Tokenization in Transformer

Sweeps over vocabulary size V and context window W on a Markov source with
BPE tokenization, measuring plateau loss in bits per source symbol.

```bash
# Train (GPU; one seed, sweeps V × W, ~2–4 h)
conda run -n bane python scripts/run_tokenization_transformer.py

# Plot curves and plateau figure
conda run -n bane python scripts/plot_tokenization_transformer.py
```

Figures are written to `outputs/tokenization_transformer/figures/`:
- `plateau_vs_W.{pdf,png}` — plateau loss vs. window size, one line per V
- `curves_W{w}_steps.{pdf,png}` — loss vs. gradient steps (one per W)
- `curves_W{w}_bits.{pdf,png}` — loss vs. source symbols seen (one per W)

### WikiText-103 Tokenizer Span Analysis

Measures the character-span distribution of W consecutive tokens for
several standard tokenizers (BPE, WordPiece, SentencePiece, GPT-2, BERT)
on WikiText-103.

```bash
# Build tokenizers and compute span distributions (CPU; ~30 min)
conda run -n bane python scripts/run_wikitext_tokenizers.py

# Plot CDF figures
conda run -n bane python scripts/plot_wikitext_tokenizers.py
```

Figures are written to `outputs/wikitext_tokenizers/figures/cdf_W{w}.{pdf,png}`.

### Markov Tokenizer Span Analysis

Same span-distribution analysis as above, but on a synthetic binary Markov-K
source using the same BPE tokenizers trained for the tokenization transformer
experiment (V=2, 4, 6, 8, 10, 15, 20).  For each W ∈ {1, …, 12}, plots the
slack curve CDF(w_s) × log₂(V) / ratio versus source-bit span w_s.

```bash
# Generate sequence, train tokenizers, compute distributions (CPU; ~10 min)
conda run -n bane python scripts/run_markov_tokenizers.py

# Plot slack figures
conda run -n bane python scripts/plot_markov_tokenizers.py
```

Figures are written to `outputs/markov_tokenizers/figures/slack_W{w}.{pdf,png}`.

---

## Repository Layout

```
core/                 Markov source utilities, theoretical quantities,
                      n-gram baseline (used across all experiments).
models/               TransformerLM (modality-agnostic LM).
tokenizer/            Tokenizer registry: bit_fragment (Theorem 1) and
                      bpe (Theorem 3, greedy/priority encoders).
                      Singular to avoid clashing with HuggingFace's
                      `tokenizers` package on import.
representations/      Decoder + (optional) tokenizer pipeline; turns raw
                      bytes from text/audio/image files into a token id
                      stream for the compression benchmarks.
datasets/             Modality-specific loaders.
experiments/<name>/   One package per experiment family.
configs/<name>/       YAML configs, one file per run variant.
scripts/              Thin CLI entry points (sys.path -> repo root).
slurm/                SLURM batch scripts.
outputs/<name>/       All generated artifacts for that experiment family.
logs/                 SLURM stdout / stderr.
```

All experiment configs carry `experiment:` and `theorem:` header fields so
artifacts self-identify; run scripts assert the `experiment:` field matches
the expected family before doing anything.

---

## Development

### Adding a new experiment family

1. Create `experiments/<exp>/{__init__.py, run.py, plot.py, ...}` with the
   experiment logic.
2. Create `configs/<exp>/<variant>.yaml` with at least
   `experiment: <exp>` and `theorem: "<paper section>"` headers.
3. Add `scripts/run_<exp>.py` (and `scripts/plot_<exp>.py` if applicable)
   with the standard ROOT path-handling preamble:

```python
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

4. Resolve all output paths against `os.path.join(ROOT, "outputs", "<exp>", ...)`.
