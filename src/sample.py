import logging
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from utils.datasets import GAP_IDX
from utils.generation import generate_sequences_with_length, post_process_sequences
from utils.lightning_modules import DirichletFlowMatchingLightning

logger = logging.getLogger("sample")

torch.set_float32_matmul_precision("high")


def _as_list(value):
    """Normalise a Hydra value (None, scalar or list) to a plain Python list."""
    if value is None:
        return None
    value = OmegaConf.to_container(value) if OmegaConf.is_config(value) else value
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _now(device) -> float:
    """Wall-clock time; waits for queued GPU work first."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _fail(message: str):
    logger.error(message)
    sys.exit(2)


def _resolve_label(module, requested) -> int:
    """Map a requested label name to its index (exact match, then case-insensitive)."""
    names = module.label_names
    requested = str(requested)
    if requested in names:
        return names.index(requested)
    lowered = [n.lower() for n in names]
    if requested.lower() in lowered:
        return lowered.index(requested.lower())
    _fail(f"Unknown label '{requested}'. Valid labels: {', '.join(names)}.")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = to_absolute_path(cfg.checkpoint)

    t0 = _now(device)
    logger.info(f"Loading checkpoint {checkpoint} on {device}")
    module = DirichletFlowMatchingLightning.load_from_checkpoint(
        checkpoint, map_location=device
    )
    module.to(device)
    module.eval()
    t_loaded = _now(device)

    if cfg.alpha_max is not None:
        logger.info(f"Overriding alpha_max: {module.cfg.alpha_max} -> {cfg.alpha_max}")
        module.cfg.alpha_max = cfg.alpha_max
        module.condflow = None  # flow tables are rebuilt for the new alpha_max
    if module.condflow is None:
        module.setup(stage="predict")
    t_setup = _now(device)

    # ---- conditioning ----------------------------------------------------
    requested_lengths = _as_list(cfg.lengths)
    requested_labels = _as_list(cfg.labels)

    if module.use_length_conditioning:
        if not requested_lengths:
            _fail("This model is length-conditioned: set `lengths=[...]`.")
        lengths = [int(length) for length in requested_lengths]
        lo, hi = module.min_ungapped_length, module.max_ungapped_length
        for length in lengths:
            if not lo <= length <= hi:
                logger.warning(
                    f"Target length {length} is outside the training range "
                    f"[{lo}, {hi}]; length control is unreliable there."
                )
    else:
        if requested_lengths:
            logger.warning("Model is not length-conditioned; `lengths` is ignored.")
        lengths = [None]

    if module.use_label_conditioning:
        if not requested_labels:
            valid = ", ".join(module.label_names or [])
            _fail(
                f"This model is label-conditioned: set `labels=[...]` (valid: {valid})."
            )
        if module.label_names is None:
            _fail(
                "This checkpoint does not contain label names; cannot resolve labels."
            )
        label_indices = [_resolve_label(module, label) for label in requested_labels]
        label_names = [module.label_names[i] for i in label_indices]
    else:
        if requested_labels:
            logger.warning("Model is not label-conditioned; `labels` is ignored.")
        label_indices, label_names = [None], [None]

    # Every (length, label) combination gets `n_sequences` samples
    target_lengths, target_label_indices, target_label_names = [], [], []
    for length in lengths:
        for idx, name in zip(label_indices, label_names):
            target_lengths += [length] * cfg.n_sequences
            target_label_indices += [idx] * cfg.n_sequences
            target_label_names += [name] * cfg.n_sequences

    logger.info(
        f"Sampling {len(target_lengths)} sequences "
        f"(lengths={lengths if module.use_length_conditioning else 'n/a'}, "
        f"labels={label_names if module.use_label_conditioning else 'n/a'}, "
        f"alpha_max={module.cfg.alpha_max})"
    )

    # ---- sampling --------------------------------------------------------
    sequences = generate_sequences_with_length(
        module,
        # Models without length conditioning ignore the values (only the count matters)
        [length if length is not None else 0 for length in target_lengths],
        target_labels=target_label_indices if module.use_label_conditioning else None,
        device=device,
        max_seq_len=module.max_seq_len,
        max_batch_size=cfg.max_batch_size,
    )
    t_generated = _now(device)
    seq_strings, ungapped_lengths = post_process_sequences(
        sequences,
        torch.tensor(
            [length if length is not None else 0 for length in target_lengths]
        ),
        use_target_lengths=False,
        gap_idx=GAP_IDX,
    )
    t_decoded = _now(device)

    # ---- output ----------------------------------------------------------
    out_path = Path(to_absolute_path(cfg.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for i, (seq, n_res) in enumerate(zip(seq_strings, ungapped_lengths)):
            header = f">{cfg.prefix}_{i}"
            if module.use_length_conditioning:
                header += f" target_length={target_lengths[i]}"
            if module.use_label_conditioning:
                header += f" target_label={target_label_names[i]}"
            header += f" ungapped_length={int(n_res)}"
            fh.write(f"{header}\n{seq if cfg.keep_gaps else seq.replace('-', '')}\n")
    t_end = _now(device)
    logger.info(f"Wrote {len(seq_strings)} sequences to {out_path}")

    logger.info(
        f"Timing (s): load={t_loaded - t0:.2f} setup={t_setup - t_loaded:.2f} "
        f"generate={t_generated - t_setup:.2f} decode={t_decoded - t_generated:.2f} "
        f"write={t_end - t_decoded:.2f} total={t_end - t0:.2f} "
        f"| {len(seq_strings) / (t_generated - t_setup):.1f} sequences/s generating "
    )


if __name__ == "__main__":
    main()
