import logging
import os
import random
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from utils.datasets import ProteinDataModule
from utils.lightning_modules import DirichletFlowMatchingLightning

# Get loggers for different purposes
app_logger = logging.getLogger(__name__)  # Main application logic
data_logger = logging.getLogger("data_processing")  # Data-related logs
model_logger = logging.getLogger("model")  # Model-specific logs

torch.set_float32_matmul_precision("high")


def set_seed(seed: int):
    """
    Seed everything for exact reproducibility across PyTorch, NumPy, random,
    and CUDA (if available). Use before any model / DataLoader / sampler instantiation.
    """
    # 1) Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) PyTorch CPU
    torch.manual_seed(seed)
    # 4) PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 5) Torch backend flags for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app_logger.info("Starting training pipeline")
    SEED = cfg.main.training.seed

    if SEED is not None:
        app_logger.info(f"Setting seed to {SEED}")
        set_seed(SEED)
    else:
        app_logger.info("No seed provided, using random seed")
        SEED = random.randint(0, 2**32 - 1)
        set_seed(SEED)

    # Set up dataset
    fasta_path = Path(to_absolute_path(cfg.main.data.fasta_path))
    data_logger.info(
        f"Setting up dataset with batch size {cfg.main.training.batch_size} and num_workers {cfg.main.data.num_workers}. "
        f"Fasta file: {fasta_path}"
    )
    datamodule = ProteinDataModule(
        fasta_path,
        data_logger,
        batch_size=cfg.main.training.batch_size,
        val_split=0.1,
        num_workers=cfg.main.data.num_workers,
        random_state=SEED,
    )
    datamodule.setup()
    data_logger.info("Dataset setup complete.")
    if cfg.model.get("use_label_conditioning", False) and datamodule.num_unlabelled:
        raise SystemExit(
            f"model.use_label_conditioning=true, but {datamodule.num_unlabelled} sequences "
            "have no label in their FASTA header (see README, 'Labels'). Label them or remove them."
        )

    # Set up model
    model_logger.info("Setting up model")
    model = DirichletFlowMatchingLightning(
        cfg.model,
        model_logger,
        alphabet_size=cfg.main.data.alphabet_size,
        lr=cfg.main.training.lr,
        gap_weight=cfg.model.gap_weight,
        use_length_conditioning=cfg.model.use_length_conditioning,
        use_label_conditioning=cfg.model.get("use_label_conditioning", False),
        num_labels=getattr(datamodule, "num_labels", None),
        # Label names ordered by label index, so sampling can request labels by name
        label_names=(
            [datamodule.idx_to_label[i] for i in range(datamodule.num_labels)]
            if getattr(datamodule, "num_labels", 0)
            else None
        ),
        max_seq_len=datamodule.msa_columns,
        min_ungapped_length=datamodule.min_length,
        max_ungapped_length=datamodule.max_length,
    )
    model_logger.info("Model setup complete")

    # Set up run name
    run_dir = f"{cfg.log_dir}_dirichlet_flow_matching"
    run_name = f"{cfg.log_dir.split('/')[-1]}_dirichlet_flow_matching"
    os.makedirs(run_dir, exist_ok=True)
    app_logger.info(f"Run directory: {run_dir}")

    # Set up logger (WandB is an optional extra; CSVLogger is the default)
    if cfg.main.logging.wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "main.logging.wandb=true requires the optional WandB extra "
                "(e.g. `uv sync --extra wandb` or `pip install syzygy[wandb]`)."
            ) from e

        tags = [
            cfg.model.model,  # Model type
            fasta_path.stem,  # Training MSA
            cfg.main.training.track_metric.split("/")[-1],  # Tracking metric
        ]
        # Add optional tags only if not empty
        if cfg.model.use_length_conditioning:
            tags.append("length_conditioning")
        if cfg.model.get("use_label_conditioning", False):
            tags.append("label_conditioning")

        lightning_logger = WandbLogger(
            project=cfg.main.logging.wandb_project,
            tags=tags,
            log_model=cfg.main.logging.wandb_log_model,
        )
    else:
        lightning_logger = CSVLogger(run_dir, name=run_name)

    app_logger.info(f"Logger setup complete. Wandb: {cfg.main.logging.wandb}")

    # Set up checkpoint directory
    checkpoint_dir = Path(to_absolute_path(cfg.main.training.checkpoint_dir)).resolve()
    os.makedirs(checkpoint_dir, exist_ok=True)
    app_logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Set up callbacks
    track_metric = cfg.main.training.track_metric
    metric_tag = track_metric.replace("/", "-").replace("_", "-")
    checkpoint_callback = ModelCheckpoint(
        monitor=track_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=checkpoint_dir,
        # e.g. epoch-07_val-perplexity-epoch-3.412.ckpt
        filename=f"epoch-{{epoch:02d}}_{metric_tag}-{{{track_metric}:.3f}}",
        auto_insert_metric_name=False,
    )
    model_logger.info("Checkpoint callback setup complete")

    if cfg.main.training.early_stopping:
        model_logger.info("Early stopping is enabled")
        early_stop_callback = EarlyStopping(
            monitor=cfg.main.training.track_metric,
            min_delta=0.00,
            patience=cfg.main.training.early_stopping_patience,
            verbose=False,
            mode="min",
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    else:
        model_logger.info("Early stopping is disabled")
        callbacks = [checkpoint_callback]

    app_logger.info(f"Run name: {run_name}")
    app_logger.info(f"Run directory: {run_dir}")
    app_logger.info("Starting training")

    # Set up trainer
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator=cfg.main.training.accelerator,
        devices=cfg.main.training.devices,
        max_epochs=cfg.main.training.max_epochs,
        callbacks=callbacks,
        logger=lightning_logger,
        gradient_clip_val=cfg.main.training.grad_clip,
        enable_progress_bar=True,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
    )

    resume_from = cfg.main.training.get("resume_from")
    if resume_from:
        resume_from = to_absolute_path(resume_from)
        app_logger.info(f"Resuming training from checkpoint: {resume_from}")
    trainer.fit(model, datamodule, ckpt_path=resume_from or None)

    best_checkpoint = checkpoint_callback.best_model_path
    app_logger.info(f"Best model checkpoint at: {best_checkpoint}")

    if cfg.main.logging.wandb:
        try:
            lightning_logger.experiment.config.update(
                {
                    "dataset_size": datamodule.dataset_length,
                    "max_seq_length": datamodule.max_length,
                    "min_seq_length": datamodule.min_length,
                    "gap_percentage": datamodule.gap_percentage,
                    "msa_columns": datamodule.msa_columns,
                    "checkpoint_path": best_checkpoint,
                },
                allow_val_change=True,
            )
        except Exception as e:
            app_logger.warning(f"Could not log dataset information to WandB: {e}")

    app_logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
