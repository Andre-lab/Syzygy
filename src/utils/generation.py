import logging
from typing import List, Optional, Tuple, Union
import numpy as np

import torch

from utils.datasets import GAP_IDX, IDX_TO_AA

logger = logging.getLogger("generation")


def get_safe_batch_size(
    target_batch_size: int,
    seq_len: int,
    alphabet_size: int,
    hidden_dim: int,
    num_heads: int,
    device: str = "cuda",
    safety_factor: float = 0.8,
) -> int:
    """
    Largest batch (up to ``target_batch_size``) that should fit in the free memory
    of ``device``. CPU runs get a fixed 128: a bigger batch does not speed them up.
    """
    if not str(device).startswith("cuda"):
        return min(target_batch_size, 128)

    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device).index or 0)
    K, L = alphabet_size, seq_len
    # Floats per sample for one flow step (no gradients are kept): ~3 [L,K,K] flow
    # temporaries, ~14 [L,D] transformer activations and, as the worst case when
    # attention is not fused, 2 attention maps [heads,L,L]. Estimated, not measured.
    floats = L * (3 * K * K + 14 * hidden_dim) + 2 * num_heads * L * L
    return max(1, min(target_batch_size, int(free_bytes * safety_factor / (4 * floats))))


def generate_sequences_with_length(
    module,
    target_lengths: Union[List[int], int, torch.Tensor],
    target_labels: Optional[Union[List[int], int, torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    verbose: bool = True,
    return_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate sequences with specified target ungapped lengths.
    Automatically splits into smaller batches if memory is limited.

    Sampling itself is performed by
    ``DirichletFlowMatchingLightning.dirichlet_flow_inference``; this function
    only handles input conversion, memory-aware batching and decoding
    (``argmax`` of the final simplex state).

    Args:
        module: Trained ``DirichletFlowMatchingLightning`` (its flow tables are
            built on demand via ``setup`` if that has not happened yet)
        target_lengths: List or tensor of desired ungapped lengths. For models
            without length conditioning the values are ignored and only the
            number of entries (= number of sequences) matters.
        target_labels: Label indices (one per sequence, or a single int), for
            models trained with label conditioning
        device: Device to run on. Defaults to the module's device.
        max_seq_len: Maximum sequence length (with gaps). If None, uses 1.5x max target length
        max_batch_size: Maximum batch size to use. If None, auto-detects safe size
        verbose: Print progress information
        return_probs: Also return the final probabilities [B, L, K]. Off by
            default: they are B*L*K floats (about 5 GB for 100,000 sequences
            of 614 columns) and ``sample.py`` does not use them.

    Returns:
        sequences: Generated sequences [B, L] (symbol indices; on the CPU for
            multi-batch runs), or ``(sequences, final_probs)`` when
            ``return_probs`` is True (also on the CPU for multi-batch runs).
    """
    if device is None:
        device = module.device
    device = str(device)

    if isinstance(target_lengths, (list, int)):
        if isinstance(target_lengths, int):
            target_lengths = [target_lengths]
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    elif not isinstance(target_lengths, torch.Tensor):
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    else:
        target_lengths = target_lengths.to(device)

    total_sequences = len(target_lengths)
    K = module.model.alphabet_size

    if target_labels is not None:
        if isinstance(target_labels, (list, int)):
            if isinstance(target_labels, int):
                target_labels = [target_labels] * total_sequences
            target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)
        elif not isinstance(target_labels, torch.Tensor):
            target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)
        else:
            target_labels = target_labels.to(device).long()

        if len(target_labels) != total_sequences:
            raise ValueError(
                f"`target_labels` length {len(target_labels)} must match "
                f"`target_lengths` length {total_sequences}."
            )

    # Determine sequence length with buffer for gaps
    if max_seq_len is None:
        max_seq_len = int(target_lengths.max().item() * 1.5)
    L = max_seq_len

    # Determine safe batch size
    if max_batch_size is None:
        max_batch_size = get_safe_batch_size(
            target_batch_size=1024,
            seq_len=L,
            alphabet_size=K,
            hidden_dim=module.cfg.hidden_dim,
            num_heads=module.cfg.num_heads,
            device=device,
        )
        if verbose:
            logger.info(f"Auto-detected safe batch size: {max_batch_size}")

    # Check if we need to split into batches
    if total_sequences <= max_batch_size:
        # Can do in one batch
        if verbose:
            logger.info(f"Generating {total_sequences} sequences in single batch")
        sequences, probs = _generate_batch(
            module, target_lengths, target_labels, device, L
        )
        return (sequences, probs) if return_probs else sequences

    # Need to split into multiple batches
    num_batches = (total_sequences + max_batch_size - 1) // max_batch_size
    if verbose:
        logger.info(f"Generating {total_sequences} sequences in {num_batches} batches")

    all_sequences = []
    all_probs = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min((batch_idx + 1) * max_batch_size, total_sequences)

        batch_lengths = target_lengths[start_idx:end_idx]
        batch_labels = (
            None if target_labels is None else target_labels[start_idx:end_idx]
        )

        if verbose:
            logger.info(
                f"Processing batch {batch_idx + 1}/{num_batches} "
                f"(sequences {start_idx + 1}-{end_idx})"
            )

        try:
            batch_sequences, batch_probs = _generate_batch(
                module, batch_lengths, batch_labels, device, L
            )
            all_sequences.append(batch_sequences.cpu())
            if return_probs:
                all_probs.append(batch_probs.cpu())

            # Clear cache after each batch
            if device != "cpu":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.critical(
                    f"\nOOM error in batch {batch_idx + 1}. Try reducing max_batch_size."
                )
                if device != "cpu":
                    torch.cuda.empty_cache()
                raise
            else:
                raise

    # Concatenate all batches
    sequences = torch.cat(all_sequences, dim=0)

    if verbose:
        logger.info(f"Successfully generated {total_sequences} sequences")

    if return_probs:
        return sequences, torch.cat(all_probs, dim=0)
    return sequences


def _generate_batch(
    module,
    target_lengths: torch.Tensor,
    target_labels: Optional[torch.Tensor],
    device: str,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Internal function to generate a single batch of sequences by running
    ``module.dirichlet_flow_inference`` and decoding the final simplex state.
    """

    # Log target lengths for debugging
    logger.info(
        f"Generating batch with target lengths: {np.unique(target_lengths.cpu().numpy())}"
    )
    logger.info(
        f"Target labels: {np.unique(target_labels.cpu().numpy()) if target_labels is not None else None}"
    )

    B = len(target_lengths)

    # The flow tables are created in LightningModule.setup()
    if module.condflow is None:
        module.setup(stage="predict")

    # dirichlet_flow_inference only uses the *shape* (and device) of `seq`
    seq_template = torch.zeros(B, L, dtype=torch.long, device=device)

    module.eval()
    with torch.no_grad():
        _, _, xt = module.dirichlet_flow_inference(
            seq_template,
            model=module.model,
            length=target_lengths,
            label=target_labels,
            return_xt=True,
        )

    # Convert to discrete sequences
    return torch.argmax(xt, dim=-1), xt


def post_process_sequences(
    sequences, target_lengths, use_target_lengths=True, gap_idx=GAP_IDX
):
    """
    Post-process generated sequences to match target ungapped lengths.

    Args:
        sequences: Generated sequences [B, L]
        target_lengths: Target ungapped lengths [B]
        gap_idx: Index of gap character

    Returns:
        processed_sequences: List of sequences as strings
        ungapped_lengths: Actual ungapped lengths achieved
    """
    B, L = sequences.shape
    processed = []
    actual_lengths = []

    for i in range(B):
        seq = sequences[i].cpu().numpy()
        target_len = target_lengths[i].item()

        # Count non-gap positions
        ungapped_positions = seq != gap_idx
        current_ungapped_len = ungapped_positions.sum()

        # If we have too many ungapped positions, we can't easily fix it
        # Just report what we got
        if current_ungapped_len > target_len and use_target_lengths:
            # Could implement trimming logic here if needed
            pass

        # Convert to string (removing padding if present)
        seq_str = "".join([IDX_TO_AA[int(idx)] for idx in seq if idx < len(IDX_TO_AA)])
        processed.append(seq_str)
        actual_lengths.append(current_ungapped_len)

    return processed, actual_lengths
