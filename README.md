# Bytes are not Enough: Tokenization for Efficient Neural Compression

Reproducibility code for the paper. Each experiment family has its own
sub-package under `experiments/`, its own configs under `configs/`, its own
script entry points under `scripts/`, and its own output directory under
`outputs/`.

## Layout

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
datasets/             Modality-specific dataset loaders (placeholder).
compressors/          Compressor ABC + sub-packages for each codec.
                      External codecs (EDPC, TRACE) are vendored under
                      compressors/<name>/upstream/, never as submodules.
experiments/<name>/   One package per experiment family.
configs/<name>/       YAML configs, one file per run variant.
scripts/              Thin CLI entry points (sys.path -> repo root).
slurm/                SLURM batch scripts.
outputs/<name>/       All generated artifacts for that experiment family.
logs/                 SLURM stdout / stderr.
```

## Experiments and how to run them

| Experiment family | Theorem | Configs | Run | Plot | Outputs |
|---|---|---|---|---|---|
| `fragmentation_ngram` | 3.1 | `configs/fragmentation_ngram/{base,Kp1}.yaml` | `python scripts/run_fragmentation_ngram.py [config]` | (plotting is part of the run) | `outputs/fragmentation_ngram/` |
| `fragmentation_ngram` (sweep) | 3.1 | sweep parameters hard-coded in the script | `python scripts/sweep_fragmentation_ngram.py` | (plotting is part of the run) | `outputs/fragmentation_ngram/` |
| `fragmentation_transformer` | 1 | `configs/fragmentation_transformer/{A,B,C}.yaml` | `python scripts/run_fragmentation_transformer.py [--smoke] [configs...]` | `python scripts/plot_fragmentation_transformer.py [A B C]` | `outputs/fragmentation_transformer/` |
| `tokenization_transformer` | 3 | `configs/tokenization_transformer/{A_greedy,A_priority}.yaml` | `python scripts/run_tokenization_transformer.py [--smoke] [configs...]` | (plotting is part of the run) | `outputs/tokenization_transformer/` |
| `compression_benchmarks` | -- | (placeholder) | (not yet implemented) | (not yet implemented) | (not yet generated) |

Configs include `experiment:` and `theorem:` header fields so artifacts
self-identify; the run scripts assert the `experiment:` field matches the
expected family before doing anything.

## Setup

```
pip install -r requirements.txt
```

The transformer experiment can run on CPU but is much faster on GPU; the
SLURM script in `slurm/fragmentation_transformer.sh` activates a CUDA
environment and submits with `--gres=gpu:1`.

### Initializing the repo

```
git init
git add .
git commit -m "Initial flat-layout refactor"
```

(The `.gitignore` already excludes `__pycache__/`, `logs/`, etc., so
`git status` after a fresh `git init` shows only intended content.)

## Adding a new vendored compressor

Each external codec lives in `compressors/<name>/` as a sub-package. Vendor
the upstream code (do **not** use git submodules -- see the rationale in the
project plan and in `THIRD_PARTY_NOTICES.md`). Procedure:

1. Clone the upstream repo, note its commit hash.
2. Copy only the source files imported by `compressors/<name>/adapter.py`
   (plus their transitive imports) into `compressors/<name>/upstream/`.
3. Copy the upstream `LICENSE` verbatim. Preserve copyright headers in the
   source files.
4. Fill in `compressors/<name>/upstream/README.md` with the upstream URL,
   commit hash, vendoring date, list of files removed, and the exact
   command(s) used to vendor.
5. Start `compressors/<name>/upstream/MODIFICATIONS.md` empty; every later
   edit gets a one-paragraph entry (file, rationale, expected effect on
   benchmark numbers).
6. Strip author-identifying boilerplate from any vendored READMEs for
   double-blind anonymization. Keep license / copyright headers in source
   files (mandatory).
7. Add a row to repo-root `THIRD_PARTY_NOTICES.md`.

Imports stay explicit so the boundary is obvious in diffs:

```python
from compressors.<name>.upstream.<module> import <Class>   # their code
from compressors.<name>.adapter import <NAME>Compressor    # our wrapper
```

The vanilla transformer compressor under `compressors/transformer/` is our
own work and has no `upstream/` directory.

## Adding a new experiment family

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
5. Add a row to the table above.
