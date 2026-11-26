import logging
import time

import lightning.pytorch as pl
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.distributions import Dirichlet
from torchmetrics.text import EditDistance

from utils.datasets import AMINO_ACID_ALPHABET, GAP_IDX, IDX_TO_AA
from utils.models import (
    AdaLNTransformerModel,
    DirichletConditionalFlowHybrid,
    expand_simplex,
    simplex_proj,
)


def _wandb_logging_active(logger) -> bool:
    """
    True only when the attached Lightning logger is a WandbLogger.

    WandB (and the plotting libraries used for WandB-only diagnostics) are an
    optional extra: they are imported lazily, only when this returns True.
    """
    return isinstance(logger, WandbLogger)


def to_sequence(seq, alphabet_size=20):
    # seq: 1D iterable of ints / torch tensor
    if alphabet_size == 20:
        return "".join([IDX_TO_AA[int(num)] for num in seq])
    elif alphabet_size == 21:
        chars = []
        for num in seq:
            n = int(num)
            if n == 20:  # gap/insert index -> ignore
                continue
            if 0 <= n < 20:
                chars.append(IDX_TO_AA[n])
            else:
                raise ValueError(f"Index {n} out of range for alphabet_size=21")
        return "".join(chars)
    else:
        raise ValueError(f"Unsupported alphabet_size: {alphabet_size}")


def flatten_cfg(cfg):
    return {k: v for k, v in cfg.items()}


def kl_divergence(p, q):
    # p, q: distributions over amino acids (length A=20)
    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum()
    return kl


def bar_plot(df, title):
    # Optional-extra imports (only reached when logging to WandB).
    import matplotlib.pyplot as plt
    import seaborn as sns

    import wandb

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(data=df, x="Amino Acid", y="Frequency", hue="Kind", ax=ax)
    ax.set_ylim(0, max(df["Frequency"].values) * 1.1)
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return wandb.Image(fig)


class DirichletFlowMatchingLightning(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        model_logger=None,  # Falls back to logging.getLogger("model") when None
        alphabet_size=20,
        num_classes=1,
        lr=5e-4,
        use_length_conditioning=True,  # Enable length conditioning
        use_label_conditioning=False,  # Optional label conditioning
        num_labels: int | None = None,  # Required if use_label_conditioning=True
        # Label names ordered by label index (saved in checkpoints so that labels
        # can be requested by name when sampling)
        label_names: list[str] | None = None,
        gap_weight=1.0,  # Weight for gap character in loss
        max_seq_len=512,  # Maximum sequence length
        min_ungapped_length=10,  # Minimum ungapped length (for length conditioning)
        max_ungapped_length=500,  # Maximum ungapped length (for length conditioning)
    ):
        super().__init__()
        flat_params = flatten_cfg(model_cfg)
        flat_params["alphabet_size"] = alphabet_size
        flat_params["num_classes"] = num_classes
        flat_params["lr"] = lr
        flat_params["use_length_conditioning"] = use_length_conditioning
        flat_params["use_label_conditioning"] = use_label_conditioning
        flat_params["num_labels"] = num_labels
        flat_params["label_names"] = label_names
        flat_params["gap_weight"] = gap_weight
        # Save hyperparameters (stored in checkpoints so they can be reloaded).
        # The logger object is not a hyperparameter, so it is excluded; it
        # defaults to logging.getLogger("model") when a checkpoint is reloaded.
        self.save_hyperparameters(ignore=["model_logger"])

        # Set parameters
        self.cfg = model_cfg
        self.K = alphabet_size
        self.num_classes = num_classes
        self.lr = lr
        self.use_length_conditioning = use_length_conditioning
        self.use_label_conditioning = use_label_conditioning
        self.num_labels = num_labels
        if label_names is not None and num_labels is not None:
            if len(label_names) != num_labels:
                raise ValueError(
                    f"`label_names` has {len(label_names)} entries but "
                    f"`num_labels` is {num_labels}."
                )
        self.label_names = list(label_names) if label_names is not None else None
        self.gap_weight = gap_weight
        self.max_seq_len = max_seq_len
        self.min_ungapped_length = min_ungapped_length
        self.max_ungapped_length = max_ungapped_length

        # Build model
        self.load_model(self.cfg.model)

        # Set other variables
        self.seq_data = None
        self.model_logger = (
            model_logger if model_logger is not None else logging.getLogger("model")
        )
        self.last_log_time = time.time()
        self.condflow = None  # Will be initialized in move_to_device
        self.position_entropy = []
        self.sequence_recovery = []
        self.levenshtein = EditDistance()

    def setup(self, stage):
        self.model_logger.info(f"Initialising conditional flow on: {self.device}")
        self.condflow = DirichletConditionalFlowHybrid(
            K=self.model.alphabet_size,
            alpha_spacing=0.001,
            alpha_max=self.cfg.alpha_max,
            device=self.device,
        )

    def compute_loss_entropy_perplexity(
        self,
        logits,
        seq,
        mask,
    ):
        """
        logits: (B, L, A)  — raw model outputs for A amino acids
        seq:    (B, L)     — integer targets in [0..A-1]
        mask:   (B, L)     — 1 for real positions, 0 for padding
        """

        # Cross‑entropy per position
        if self.model.alphabet_size == 21:
            # Weight gaps less than other amino acids
            class_weights = torch.ones(self.model.alphabet_size).to(self.device)
            class_weights[GAP_IDX] = self.gap_weight  # e.g., 0.1
            ce_losses = F.cross_entropy(
                logits.transpose(1, 2), seq, reduction="none", weight=class_weights
            )
        else:
            ce_losses = F.cross_entropy(
                logits.transpose(1, 2),  # → (B, A, L)
                seq,  # → (B, L)
                reduction="none",  # → (B, L)
            )
        ce_loss = (ce_losses * mask).sum() / mask.sum()

        # Per‑position entropy: H(p) = -∑ p log p
        probs = F.softmax(logits, dim=-1)  # (B, L, A)
        log_probs = torch.log(probs + 1e-12)
        entropy_per_pos = -(probs * log_probs).sum(dim=-1)  # (B, L)
        entropy = (entropy_per_pos * mask).sum() / mask.sum()

        # Batch-marginal entropy
        # sum probs over batch+positions, then normalize
        weighted = (probs * mask.unsqueeze(-1)).sum(dim=(0, 1))  # (A,)
        p_bar = weighted / weighted.sum()
        batch_ent = -(p_bar * torch.log(p_bar + 1e-12)).sum()

        # Compute perplexity from CE
        perplexity = torch.exp(ce_loss)

        return ce_loss, entropy, perplexity, batch_ent

    def amino_acid_distribution(self, seq, mask):
        # seq: (B, L), mask: (B, L)
        masked_seq = seq[mask.bool()]
        counts = torch.bincount(masked_seq, minlength=self.model.alphabet_size).float()
        dist = counts / counts.sum()
        return dist

    def log_aa_distributions(self, seq, pred_seq, mask, split):
        # seq, pred_seq, mask: (B, L)
        p_true = self.amino_acid_distribution(seq, mask)
        p_pred = self.amino_acid_distribution(pred_seq, mask)
        kl = kl_divergence(p_true, p_pred)

        # Log histogram via wandb directly (WandB is an optional extra)
        if _wandb_logging_active(self.logger):
            import matplotlib.pyplot as plt
            import pandas as pd

            import wandb

            # Create a DataFrame for plotting
            temp_df = pd.DataFrame(
                {
                    "Frequency": np.concatenate(
                        [p_true.cpu().numpy(), p_pred.cpu().numpy()]
                    ),
                    "Kind": (
                        ["True"] * len(p_true.cpu().numpy())
                        + ["Predicted"] * len(p_pred.cpu().numpy())
                    ),
                    "Amino Acid": np.concatenate(
                        [
                            list(AMINO_ACID_ALPHABET[: self.model.alphabet_size]),
                            list(AMINO_ACID_ALPHABET[: self.model.alphabet_size]),
                        ]
                    ),
                }
            )

            self.logger.experiment.log(
                {
                    f"{split}/aa_freq_true": wandb.Histogram(p_true.cpu().numpy()),
                    f"{split}/aa_freq_pred": wandb.Histogram(p_pred.cpu().numpy()),
                    f"{split}/aa_freq_img": bar_plot(
                        temp_df, f"{split.capitalize()} AA Frequency"
                    ),
                },
                # step=self.global_step,
            )
            plt.close("all")  # Close all plots to avoid memory issues

        return kl

    def log_per_position_heatmap(self, logits, mask, split="train"):
        """
        logits: (B, L, 20) — model outputs
        mask:   (B, L) — binary padding mask

        WandB-only diagnostic: does nothing unless the attached logger is a
        WandbLogger (WandB and matplotlib are optional extras).
        """
        if not _wandb_logging_active(self.logger):
            return

        import matplotlib.pyplot as plt

        import wandb

        B, L, A = logits.shape

        probs = F.softmax(logits, dim=-1)  # (B, L, A)

        # Mask: (B, L, 1) → broadcast to (B, L, A)
        probs = probs * mask.unsqueeze(-1)

        # Average over batch dimension, accounting for masking
        masked_counts = mask.sum(dim=0).unsqueeze(-1).clamp(min=1)  # (L, 1)
        mean_probs = probs.sum(dim=0) / masked_counts  # (L, A)
        mean_probs = mean_probs.transpose(0, 1).detach().cpu().numpy()  # (A, L)

        # Plot as heatmap
        fig, ax = plt.subplots(figsize=(L / 4, 4))
        im = ax.imshow(
            mean_probs, aspect="auto", cmap="viridis", interpolation="nearest"
        )

        ax.set_yticks(range(A))
        ax.set_yticklabels(list(AMINO_ACID_ALPHABET[: self.model.alphabet_size]))
        ax.set_xlabel("Position")
        ax.set_title(f"{split.capitalize()} AA Probabilities")

        fig.colorbar(im, ax=ax, label="Probability")
        self.logger.log_image(
            key=f"{split}/aa_prob_heatmap",
            images=[wandb.Image(fig)],
            # step=self.global_step,
        )
        plt.close(fig)

    def conditional_sample(self, seq: torch.Tensor):
        """
        Function to conditionally sample probability path.
        seq: sequence of interest
        """
        B, L = seq.shape
        seq_one_hot = torch.nn.functional.one_hot(
            seq, num_classes=self.model.alphabet_size
        )
        alphas = (
            torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * self.cfg.alpha_scale)
            .to(seq.device)
            .float()
        )

        alphas_ = torch.ones(B, L, self.model.alphabet_size, device=seq.device)
        alphas_ = alphas_ + seq_one_hot * (alphas[:, None, None] - 1)
        xt = Dirichlet(alphas_).sample()

        return xt, alphas

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            seq, mask, length = batch
            label = None
        elif len(batch) == 4:
            seq, mask, length, label = batch
        else:
            raise ValueError(
                f"Unexpected batch format of length {len(batch)}; expected 3 or 4."
            )
        B, L = seq.shape

        # Sample path
        xt, alphas = self.conditional_sample(seq)

        # Expand simplex
        xt_inp, prior_weights = expand_simplex(xt, alphas, self.cfg.prior_pseudocount)
        self.log("prior_weight", prior_weights.mean())

        # Get logits from model
        model_kwargs = {"t": alphas}
        if self.use_length_conditioning:
            model_kwargs["length"] = length
        if self.use_label_conditioning:
            if label is None:
                raise ValueError(
                    "`use_label_conditioning=True` but dataset did not provide `label`."
                )
            model_kwargs["label"] = label
        logits = self.model(xt_inp, **model_kwargs)
        # Log heatmap only every N steps (no-op unless logging to WandB)
        log_every = max(1, getattr(self.trainer, "log_every_n_steps", 10))
        if self.global_step % log_every == 0:
            self.log_per_position_heatmap(logits, mask, split="train")

        # Compute L2 norm
        norms = torch.norm(logits, dim=-1)  # (B, L) - L2 norm per residue
        mean_norm = norms.mean().item()

        # Per position entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum(dim=-1)  # (B, L)
        self.position_entropy.append(
            torch.flatten(entropy).detach().cpu().numpy()
        )  # Average over batch

        # Calculate loss (reduces to cross-entropy)
        loss, mean_entropy, perplexity, batch_ent = (
            self.compute_loss_entropy_perplexity(logits, seq, mask)
        )
        total_loss = loss

        # Calculate duration
        current_time = time.time()
        duration = current_time - self.last_log_time
        self.last_log_time = current_time

        # Log metrics - Lightning automatically aggregates epoch values
        self.log(
            "train/loss",
            total_loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        self.log("train/ce_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        self.log(
            "train/batch_ent",
            batch_ent,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/perplexity",
            perplexity,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/alpha",
            alphas.mean(),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/prior_weight",
            prior_weights.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/duration",
            torch.tensor(duration, device=self.device),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )
        self.log(
            "train/mean_velocity_norm",
            mean_norm,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/per_position_entropy",
            mean_entropy,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        # Log mean target length
        self.log(
            "train/mean_target_length",
            length.float().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            seq, mask, length = batch
            label = None
        elif len(batch) == 4:
            seq, mask, length, label = batch
        else:
            raise ValueError(
                f"Unexpected batch format of length {len(batch)}; expected 3 or 4."
            )
        B, L = seq.shape

        # Store for gap analysis in logging
        self._last_batch_seq = seq
        self._last_batch_mask = mask

        # Pass length to inference
        logits_pred, _ = self.dirichlet_flow_inference(
            seq, model=self.model, length=length, label=label
        )
        seq_pred = torch.argmax(logits_pred, dim=-1)

        kl = self.log_aa_distributions(seq, seq_pred, mask, split="val")
        self.log(
            "val/aa_KL_divergence", kl, sync_dist=True, on_step=True, on_epoch=True
        )

        loss, mean_entropy, perplexity, batch_ent = (
            self.compute_loss_entropy_perplexity(logits_pred, seq, mask)
        )

        recovery = (seq_pred == seq).float() * mask  # Only compare valid positions

        # Compute per-sequence recovery (fraction of correctly predicted, unmasked positions per sequence)
        seq_correct = ((seq_pred == seq) * mask).sum(dim=1)
        seq_total = mask.sum(dim=1).clamp(min=1)  # avoid division by zero
        seq_recovery = seq_correct.float() / seq_total.float()  # shape: (B,)
        self.sequence_recovery.append(seq_recovery.detach().cpu().numpy())

        recovery = recovery.sum() / mask.sum()

        # Calculate levenshtein distance
        seq_pred_masked = [
            to_sequence(seq_pred[i][mask[i]], alphabet_size=self.model.alphabet_size)
            for i in range(len(seq_pred))
        ]
        seq_masked = [
            to_sequence(seq[i][mask[i]], alphabet_size=self.model.alphabet_size)
            for i in range(len(seq))
        ]
        levenshtein = self.levenshtein(seq_pred_masked, seq_masked)

        # Compute ungapped length accuracy
        seq_pred_ungapped_lengths = torch.tensor(
            [(seq_pred[i][mask[i]] != GAP_IDX).sum().item() for i in range(B)],
            device=self.device,
        )
        length_error = torch.abs(seq_pred_ungapped_lengths - length).float().mean()

        # Calculate duration
        current_time = time.time()
        duration = current_time - self.last_log_time
        self.last_log_time = current_time

        # Log metrics - Lightning automatically aggregates epoch values
        self.log(
            "val/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True
        )
        self.log(
            "val/per_position_entropy",
            mean_entropy,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val/batch_ent", batch_ent, sync_dist=True, on_step=True, on_epoch=True
        )
        self.log(
            "val/perplexity",
            perplexity,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val/recovery",
            recovery,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/duration",
            torch.tensor(duration, device=self.device),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )
        self.log(
            "val/levenshtein_distance",
            levenshtein,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        # Log length error
        self.log(
            "val/length_error",
            length_error,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/mean_target_length",
            length.float().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Log sequences to wandb (if wandb is enabled)
        if self.seq_data is None:
            # Choose 5 random sequences to log
            random_five_indices = torch.randperm(len(seq))[:5]
            seq = seq[random_five_indices]
            seq_pred = seq_pred[random_five_indices]
            mask = mask[random_five_indices]
            length_subset = length[random_five_indices]

            self.seq_data = [
                [
                    to_sequence(s[m.bool()], alphabet_size=self.model.alphabet_size),
                    to_sequence(ps[m.bool()], alphabet_size=self.model.alphabet_size),
                    str(self.current_epoch),
                    str(l.item()),  # NEW: Add target length
                    str(
                        (ps[m.bool()] != GAP_IDX).sum().item()
                    ),  # NEW: Add predicted ungapped length
                ]
                for s, ps, m, l in zip(
                    seq,
                    seq_pred,
                    mask,
                    length_subset,
                )
            ]

        return loss

    def on_train_epoch_end(self):
        # Print training epoch summary
        self.model_logger.info(
            f"Epoch {self.current_epoch} training summary: "
            f"Loss: {self.trainer.callback_metrics['train/loss']:.4f}, "
            f"Perplexity: {self.trainer.callback_metrics['train/perplexity']:.4f}, "
            f"Alpha: {self.trainer.callback_metrics['train/alpha']:.4f}, "
            f"Prior Weight: {self.trainer.callback_metrics['train/prior_weight']:.4f}, "
            f"Duration: {self.trainer.callback_metrics['train/duration']:.2f}s"
        )

        self.position_entropy = np.concatenate(self.position_entropy)

        # WandB-only histogram (WandB is an optional extra)
        if _wandb_logging_active(self.logger):
            try:
                import wandb

                self.logger.experiment.log(
                    {"train/entropy_histogram": wandb.Histogram(self.position_entropy)}
                )
            except Exception as e:
                self.model_logger.error(f"Error logging entropy histogram: {e}")

        self.position_entropy = []

    def on_validation_epoch_end(self):
        # Log the sequences (only for loggers that support text tables, i.e. WandB)
        if _wandb_logging_active(self.logger):
            try:
                self.logger.log_text(
                    key="sample_sequences",
                    columns=[
                        "Real sequence",
                        "Predicted sequence",
                        "Epoch",
                        "Target Length",
                        "Predicted Length",
                    ],
                    data=self.seq_data,
                )
            except Exception as e:
                self.model_logger.error(
                    f"Error logging sequences: {e}. Ensure that the logger supports logging text data."
                )
            self.model_logger.info(
                f"Epoch {self.current_epoch} validation sequences logged."
            )
        self.seq_data = None  # Reset after logging
        # Print validation epoch summary
        self.model_logger.info(
            f"Epoch {self.current_epoch} validation summary: "
            f"Loss: {self.trainer.callback_metrics['val/loss']:.4f}, "
            f"Perplexity: {self.trainer.callback_metrics['val/perplexity']:.4f}, "
            f"Recovery: {self.trainer.callback_metrics['val/recovery']:.4f}, "
            f"Duration: {self.trainer.callback_metrics['val/duration']:.2f}s"
        )

        self.sequence_recovery = np.concatenate(self.sequence_recovery)

        # WandB-only histogram (WandB is an optional extra)
        if _wandb_logging_active(self.logger):
            try:
                import wandb

                self.logger.experiment.log(
                    {"val/recovery_histogram": wandb.Histogram(self.sequence_recovery)}
                )
            except Exception as e:
                self.model_logger.error(f"Error logging recovery histogram: {e}")

        self.sequence_recovery = []

    def dirichlet_flow_inference(
        self, seq, model, length=None, label=None, return_xt=False
    ):
        """
        Perform inference using Dirichlet flow matching.

        This is the single Euler integration of the probability-flow ODE (from
        alpha=1 to ``cfg.alpha_max``) used both for validation during training
        and for sampling (see ``utils.generation``).

        Args:
            seq: Input sequence (B, L). Only its shape is used.
            model: The model to use for inference
            length: Target ungapped lengths (B,)
            label: Target label indices (B,), when label conditioning is used
            return_xt: If True, additionally return the final simplex state
                ``xt`` so that sequences can be decoded as ``argmax(xt)``.

        Returns:
            ``(logits, x0)`` -- the model logits of the last integration step and
            the prior sample -- or ``(logits, x0, xt)`` when ``return_xt`` is True.
        """
        B, L = seq.shape
        K = model.alphabet_size
        x0 = torch.distributions.Dirichlet(
            torch.ones(B, L, model.alphabet_size, device=seq.device)
        ).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        t_span = torch.linspace(
            1, self.cfg.alpha_max, self.cfg.num_integration_steps, device=self.device
        )
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            # Expand simplex
            xt_expanded, prior_weights = expand_simplex(
                xt, s[None].expand(B), self.cfg.prior_pseudocount
            )

            # Get logits from model (pass length if using length conditioning)
            model_kwargs = {"t": s[None].expand(B)}
            if self.use_length_conditioning and length is not None:
                model_kwargs["length"] = length
            if self.use_label_conditioning:
                if label is None:
                    raise ValueError(
                        "`use_label_conditioning=True` but no `label` was provided for inference."
                    )
                model_kwargs["label"] = label

            logits = model(xt_expanded, **model_kwargs)

            flow_probs = torch.nn.functional.softmax(
                logits / self.cfg.flow_temp, -1
            )  # [B, L, K]

            if (
                not torch.allclose(
                    flow_probs.sum(2), torch.ones((B, L), device=self.device), atol=1e-4
                )
                or not (flow_probs >= 0).all()
            ):
                self.model_logger.warning(
                    f"WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs < 0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex."
                )
                flow_probs = simplex_proj(flow_probs)

            # Calculate conditional flow factor
            c_factor = self.condflow.c_factor(xt, s)

            if torch.isnan(c_factor).any():
                self.model_logger.critical(
                    f"NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}"
                )
                raise RuntimeError(
                    f"NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}"
                )

            if not (flow_probs >= 0).all():
                self.model_logger.debug(f"flow_probs.min(): {flow_probs.min()}")

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)

            if (
                not torch.allclose(
                    xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4
                )
                or not (xt >= 0).all()
            ):
                self.model_logger.warning(
                    f"WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt < 0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex."
                )
                xt = simplex_proj(xt)

        if return_xt:
            return logits, x0, xt
        return logits, x0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def load_model(self, model_type):
        if model_type == "transformer_adaln":
            self.model = AdaLNTransformerModel(
                self.cfg,
                alphabet_size=self.K,
                num_cls=self.num_classes,
                use_length_conditioning=self.use_length_conditioning,
                use_label_conditioning=self.use_label_conditioning,
                num_labels=self.num_labels,
                min_ungapped_length=self.min_ungapped_length,
                max_ungapped_length=self.max_ungapped_length,
                max_seq_len=self.max_seq_len,
            )
        else:
            self.model_logger.critical(f"{model_type} not implemented yet.")
            raise NotImplementedError()
