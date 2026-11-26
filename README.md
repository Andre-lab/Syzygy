# Syzygy

A lightweight and conditionable Dirichlet flow matching model for the generation of artificial homologues and new members of existing protein families.

Syzygy is trained on a single multiple sequence alignment (MSA) and generates new sequences of that family directly in alignment space. Generation can optionally be steered by a target sequence length and/or a class label (for example the fluorescence colour of a GFP).

## Contents

- [Capabilities](#capabilities)
- [Installation](#installation)
- [Preparing your data](#preparing-your-data)
- [Training](#training)
- [Sampling](#sampling)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Capabilities

- Family-specific generative model. One model is trained per protein family, from an aligned FASTA file. Training takes the alignment as-is; no structures, language-model embeddings or external databases are needed.
- Dirichlet flow matching on the simplex. Each alignment column is modelled as a distribution over 21 symbols (20 amino acids and the gap `-`). Sampling integrates the learned flow (100 Euler steps by default) from the prior to a sequence. A small transformer with AdaLN-Zero conditioning (4 layers, hidden size 256, 8 heads by default) predicts the denoised distribution at each step.
- Gap-aware. Because generation happens on the alignment width, the model learns where the family places its gaps. Output gaps are removed by default; `keep_gaps=true` writes the aligned rows instead.
- Optional conditioning, chosen at training time and used at sampling time:

  | Mode | Training option | Sampling option |
  | --- | --- | --- |
  | Unconditional | (default) | none |
  | Length conditioning | `model.use_length_conditioning=true` | `'lengths=[230,250]'` |
  | Label conditioning | `model.use_label_conditioning=true` | `'labels=[green,yellow]'` |
  | Both | both options above | both options above, every length × label combination is sampled |

- Lightweight. The model trains and samples on a single GPU, and also runs on CPU (training and especially validation are much slower there).
- Reproducible. Seeded training and sampling, Hydra configs for every option, checkpoints that store everything needed to sample (including label names).
- Optional Weights & Biases logging. Off by default; metrics are written to CSV files unless you enable it.

## Installation

Syzygy is managed with [uv](https://docs.astral.sh/uv/). It requires Python 3.12 (uv will fetch it for you if needed) and supports Linux and macOS. On macOS, PyTorch is pinned to 2.4.0; use Linux with an NVIDIA GPU for fast training.

```bash
git clone https://github.com/Andre-lab/Syzygy.git
cd Syzygy
uv sync
```

This creates `.venv/` and installs the pinned dependencies from `uv.lock`. Run every command below with `uv run ...` (or activate the environment with `source .venv/bin/activate`).

### Optional: Weights & Biases

```bash
uv sync --extra wandb
uv run wandb login
```

The `wandb` extra installs `wandb` plus the plotting libraries (`matplotlib`, `pandas`, `seaborn`) used for the diagnostic figures that are only logged to W&B. Without it, the code never imports these packages.

## Preparing your data

Training needs one aligned FASTA file (an MSA in which every sequence has the same number of columns):

- Only the 20 standard amino acids and `-` are accepted (case-insensitive). Sequences containing any other character (for example `X`, `B`, `Z`, `U`) are skipped, and the number skipped is logged. `.` (Pfam/HMMER gap) is read as `-`.
- The ungapped length of each sequence is computed automatically; it is the quantity used for length conditioning.
- 10% of the sequences are held out for validation (stratified by rare amino acids).

Data is not shipped with the repository. `data/` is git-ignored, so put your alignments there (or anywhere else) and point the training command at them.

### Labels (only for label conditioning)

Labels are read from the FASTA header. The label is the token following one of these keys, up to the next space, `;` or `|`. Keys are case-insensitive and are checked in this order:

1. `color=`, `colour=`, `fluorescence=` or `fluor=`
2. `label=` or `label:`

```text
>YDB82650.1:1-238 sfGFP-elastin-like fusion protein [synthetic construct] | colour=green
>XYZ12345.1:1-238 yellow fluorescent protein variant | label=yellow
```

The distinct label names found in the file are sorted alphabetically and stored in the checkpoint, so at sampling time you request labels by name. If you enable label conditioning, every sequence must be labelled: training stops at startup and reports how many sequences have no label.

## Training

Training is a [Hydra](https://hydra.cc/) application: `src/train.py` with the configs in `src/conf/`. Defaults follow the paper settings (batch size 128, learning rate 5e-4, up to 1000 epochs, early stopping with patience 10 on validation perplexity, seed 42).

Minimal, unconditional run:

```bash
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta
```

Conditioned models:

```bash
# length-conditioned
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta \
    model.use_length_conditioning=true

# label-conditioned (needs labels in the FASTA headers)
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta \
    model.use_label_conditioning=true

# both
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta \
    model.use_length_conditioning=true model.use_label_conditioning=true
```

Useful options (any key in `src/conf/` can be overridden the same way):

| Override | Meaning | Default |
| --- | --- | --- |
| `main.data.fasta_path` | Training MSA (required) | none |
| `main.training.accelerator` / `main.training.devices` | e.g. `cpu`, `gpu` / `1` | `auto` / `auto` |
| `main.data.num_workers` | Data loader workers (use `0` if workers cause problems) | `4` |
| `main.training.batch_size` | Batch size | `128` |
| `main.training.max_epochs` | Maximum epochs | `1000` |
| `main.training.lr` | Learning rate | `5e-4` |
| `main.training.seed` | Random seed | `42` |
| `main.training.early_stopping` | Enable early stopping | `true` |
| `main.training.checkpoint_dir` | Where checkpoints are written | `checkpoints/` |
| `main.training.resume_from` | Resume from a `.ckpt` file | `null` |
| `model.num_layers`, `model.hidden_dim`, `model.num_heads` | Transformer size | `4`, `256`, `8` |
| `model.alpha_max` | Final Dirichlet concentration (flow tables and sampling endpoint) | `12` |
| `model.gap_weight` | Cross-entropy weight of the gap class (`1` = unweighted) | `1` |

### Outputs

- `checkpoints/` holds the best checkpoint by validation perplexity, named like `epoch-07_val-perplexity-epoch-3.412.ckpt`, and `last.ckpt`. When a file already exists Lightning adds a suffix (`last-v1.ckpt`), so use a separate `main.training.checkpoint_dir` for each model.
- `logs/<timestamp>_dirichlet_flow_matching/` holds the CSV metrics when W&B is off.
- All of these directories are git-ignored.

### Resuming

```bash
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta \
    main.training.resume_from=checkpoints/last.ckpt
```

Use the same data and model options as the original run.

### Logging to Weights & Biases

After `uv sync --extra wandb` and `wandb login`:

```bash
uv run src/train.py main.data.fasta_path=data/my_family_aligned.fasta \
    main.logging.wandb=true main.logging.wandb_project=my-project
```

Add `main.logging.wandb_log_model=true` to upload checkpoints as W&B artifacts.

## Sampling

`src/sample.py` loads a checkpoint written by training and writes a FASTA file. It is also a Hydra app, configured in `src/conf/sample.yaml`. The model's conditioning mode is read from the checkpoint, so you only need to supply the targets it requires.

```bash
# unconditional
uv run src/sample.py \
    checkpoint=checkpoints/last.ckpt \
    output=samples.fasta \
    n_sequences=1000

# length-conditioned: 1000 sequences for each requested ungapped length
uv run src/sample.py checkpoint=checkpoints/last.ckpt output=samples.fasta \
    n_sequences=1000 'lengths=[230,250]'

# label-conditioned: labels are requested by name, as in the training FASTA headers
uv run src/sample.py checkpoint=checkpoints/last.ckpt output=samples.fasta \
    n_sequences=1000 'labels=[green,yellow]'

# co-conditioned: n_sequences for every length x label combination (here 4 x 100)
uv run src/sample.py checkpoint=checkpoints/last.ckpt output=samples.fasta \
    n_sequences=100 'lengths=[230,250]' 'labels=[green,yellow]'
```

Always put the quotes around `'lengths=[...]'` and `'labels=[...]'`; otherwise shells such as zsh try to expand the brackets.

| Option | Meaning | Default |
| --- | --- | --- |
| `checkpoint` | Checkpoint to sample from (required) | none |
| `output` | Output FASTA path (required) | none |
| `n_sequences` | Sequences per (length, label) combination | `100` |
| `lengths` | Target ungapped length(s); required for length-conditioned models | `null` |
| `labels` | Target label name(s); required for label-conditioned models | `null` |
| `alpha_max` | Override the sampling endpoint stored in the checkpoint | `null` |
| `seed` | Sampling seed | `null` |
| `device` | e.g. `cpu`, `cuda:0` | `cuda:0` if available, else `cpu` |
| `max_batch_size` | Largest sampling batch; estimated from free GPU memory if unset | `null` |
| `keep_gaps` | Write aligned rows (with `-`) instead of ungapped sequences | `false` |
| `prefix` | FASTA header prefix | `sample` |

Each header records the requested targets and the length actually produced:

```text
>sample_0 target_length=230 target_label=green ungapped_length=228
```

Requested lengths outside the range seen in training trigger a warning, because length control is unreliable there. An unknown label name stops the run and lists the valid names. Label names are matched exactly first, then case-insensitively.

## Repository layout

```text
src/
├── train.py              # training entry point (Hydra)
├── sample.py             # sampling entry point (Hydra)
├── conf/
│   ├── config.yaml       # training defaults
│   ├── sample.yaml       # sampling defaults
│   ├── main/default.yaml # data + training options
│   ├── model/default.yaml# architecture + flow options
│   └── hydra/            # colour-coded console logging for sampling
└── utils/
    ├── models.py         # AdaLN-Zero transformer, flow tables, simplex utilities
    ├── lightning_modules.py  # Lightning module: training, validation, Dirichlet flow inference
    ├── datasets.py       # MSA loading, label parsing, data module
    └── generation.py     # batched sequence generation and post-processing
```

## Troubleshooting

- **`zsh: no matches found: lengths=[230,250]`**: quote list overrides: `'lengths=[230,250]'`.
- **Hydra error `mismatched input '=' expecting <EOF>`**: Hydra does not accept a bare `=` inside a value. This only affects checkpoints saved by older versions, whose names looked like `epoch=00-val_perplexity_epoch=8.414.ckpt`. Rename the file, or quote it twice: `'checkpoint="checkpoints/epoch=00-val_perplexity_epoch=8.414.ckpt"'`. Current checkpoint names contain no `=`.
- **`This model is length-conditioned: set lengths=[...]`** (or the same for labels): the checkpoint was trained with that conditioning, so sampling must specify it.
- **`main.logging.wandb=true requires the optional WandB extra`**: run `uv sync --extra wandb`.
- **uv cannot resolve or install**: the project requires Python 3.12 (`requires-python = ">=3.12,<3.13"`), because the macOS PyTorch pin only has 3.12 wheels.
- **Out-of-memory when sampling**: set `max_batch_size` to a smaller value.

## Citation

The accompanying paper has not been submitted yet. Citation details will be added here once it is available. Until then, please refer to this repository: <https://github.com/Andre-lab/Syzygy>.

<!--
Citation placeholder (to be completed on submission):

@article{syzygy,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD},
  doi     = {TBD}
}
-->

## License

Released under the [MIT License](LICENSE).
