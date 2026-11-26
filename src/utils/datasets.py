import numpy as np
import lightning.pytorch as pl
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACID_ALPHABET)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
GAP_IDX = AA_TO_IDX["-"]
PAD_TOKEN = len(AMINO_ACID_ALPHABET) + 5  # Make sure it is not in the alphabet


def _parse_label_from_fasta_description(description: str):
    """
    Best-effort parsing of a conditioning label from a FASTA header/description.

    Supported patterns (case-insensitive, order-agnostic):
      - "color=<label>" or "colour=<label>" (e.g. the fluorescence colour of a GFP)
      - "fluorescence=<label>" or "fluor=<label>"
      - "label=<label>" or "label:<label>"

    Returns:
        str | None
    """
    if not description:
        return None

    desc = description.strip()
    lower = desc.lower()
    keys = ("color=", "colour=", "fluorescence=", "fluor=")
    for k in keys:
        if k in lower:
            start = lower.index(k) + len(k)
            # Parse until whitespace or semicolon or pipe (common FASTA separators)
            end = start
            while end < len(desc) and desc[end] not in (" ", "\t", ";", "|"):
                end += 1
            label = desc[start:end].strip()
            return label if label else None

    # Fallback: attempt "label:<something>"
    if "label:" in lower or "label=" in lower:
        if "label:" in lower:
            start = lower.index("label:") + len("label:")
        else:
            start = lower.index("label=") + len("label=")
        end = start
        while end < len(desc) and desc[end] not in (" ", "\t", ";", "|"):
            end += 1
        label = desc[start:end].strip()
        return label if label else None

    return None


def pad_collate(batch):
    """Pads a list of variable length Tensors and returns sequences, masks, and lengths.

    If items also include a `label` key, a 4-tuple is returned:
        (padded, mask, lengths, label_ids)
    """
    sequences = [item["sequence"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels_list = [item.get("label", None) for item in batch]

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN)

    # Create mask for valid positions (True for real sequence positions)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        # Find positions where sequence is not gap or padding
        mask[i, ~(seq == PAD_TOKEN)] = 1
        # Set mask for padding positions to 0
        mask[i, seq == PAD_TOKEN] = 0

    # Create gap mask (True for gap positions, excluding padding)
    gap_mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        gap_positions = (seq == GAP_IDX).nonzero(as_tuple=True)[0]
        gap_mask[i, gap_positions] = 1

    if any(c is not None for c in labels_list):
        label_ids = torch.tensor(
            [(-1 if c is None else int(c)) for c in labels_list], dtype=torch.long
        )
        return padded, mask, lengths, label_ids

    return padded, mask, lengths


class ProteinSequenceDataset(Dataset):
    def __init__(self, fasta_path):
        """
        Args:
            fasta_path: Path to MSA fasta file
        """
        self.sequences = []
        self.lengths = []  # Store ungapped lengths
        self.total_gap_count = 0
        self.aa_presence = []  # Track which AAs are present in each sequence
        self.labels = []  # Raw string labels or None per sequence
        self.label_ids = []  # Integer-encoded labels (or None) per sequence
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.n_dropped = 0  # Sequences skipped because of characters outside the alphabet

        for record in SeqIO.parse(fasta_path, "fasta"):
            # Pfam/HMMER alignments use "." for gap columns
            seq_str = str(record.seq).upper().replace(".", "-")

            # Validate all characters are in alphabet
            if not all(c in AA_TO_IDX for c in seq_str):
                self.n_dropped += 1
            else:
                label_name = _parse_label_from_fasta_description(
                    getattr(record, "description", "") or ""
                )

                # Calculate gap count
                gap_count = seq_str.count("-")

                encoded = [AA_TO_IDX[c] for c in seq_str]
                tensor_seq = torch.tensor(encoded, dtype=torch.long)

                # Store ungapped length (biological sequence length)
                ungapped_length = len(seq_str) - gap_count

                # Track amino acid presence (excluding gaps)
                aa_present = set(seq_str) - {"-"}
                self.aa_presence.append(aa_present)

                self.sequences.append(tensor_seq)
                self.lengths.append(ungapped_length)
                self.total_gap_count += gap_count
                self.labels.append(label_name)

        # Build a stable mapping for labels if present (indices follow the
        # alphabetical order of the label names)
        unique_labels = sorted({c for c in self.labels if c is not None})
        if unique_labels:
            self.label_to_idx = {c: i for i, c in enumerate(unique_labels)}
            self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
            self.label_ids = [
                (None if c is None else self.label_to_idx[c]) for c in self.labels
            ]
        else:
            self.label_ids = [None for _ in self.labels]

        # Store IQR range for lengths for potential use in stratification
        if self.lengths:
            self.p10 = float(np.percentile(self.lengths, 10))
            self.p25 = float(np.percentile(self.lengths, 25))
            self.p75 = float(np.percentile(self.lengths, 75))
            self.p90 = float(np.percentile(self.lengths, 90))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequences[idx],
            "length": self.lengths[idx],  # Ungapped length
        }
        if self.label_ids[idx] is not None:
            item["label"] = self.label_ids[idx]
        return item

    def get_stratification_labels(self):
        """
        Create stratification labels based on rare amino acid presence.
        Returns a list of string labels for each sequence.
        """
        # Count amino acid frequencies across all sequences
        aa_counts = {aa: 0 for aa in AMINO_ACID_ALPHABET if aa != "-"}
        for aa_set in self.aa_presence:
            for aa in aa_set:
                aa_counts[aa] += 1

        # Identify rare amino acids (present in < 20% of sequences)
        total_seqs = len(self.sequences)
        rare_threshold = 0.2 * total_seqs
        rare_aas = {aa for aa, count in aa_counts.items() if count < rare_threshold}

        # Create labels based on which rare AAs are present
        labels = []
        for aa_set in self.aa_presence:
            rare_present = sorted(aa_set & rare_aas)
            if rare_present:
                # Create label from rare AAs present (e.g., "W_Y" if W and Y present)
                label = "_".join(rare_present)
            else:
                label = "common_only"
            labels.append(label)

        return labels


class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fasta_path,
        data_logger,
        batch_size=32,
        val_split=0.1,
        num_workers=4,
        random_state=42,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.data_logger = data_logger
        self.random_state = random_state

    def setup(self, stage=None):
        dataset = ProteinSequenceDataset(self.fasta_path)

        # Get stratification labels
        strat_labels = dataset.get_stratification_labels()

        # Perform stratified split
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_split,
            stratify=strat_labels,
            random_state=self.random_state,
        )

        self.train_ds = Subset(dataset, train_idx)
        self.val_ds = Subset(dataset, val_idx)

        # Log statistics
        all_lengths = [dataset.lengths[i] for i in range(len(dataset))]
        self.data_logger.info(
            f"Dataset sizes: train={len(self.train_ds)}, val={len(self.val_ds)}"
        )
        if dataset.n_dropped:
            self.data_logger.warning(
                f"Dropped {dataset.n_dropped} of {dataset.n_dropped + len(dataset)} "
                "sequences containing characters other than the 20 amino acids and '-'/'.'"
            )
        self.data_logger.info(
            f"Ungapped length range: {min(all_lengths)}-{max(all_lengths)}, "
            f"mean={sum(all_lengths) / len(all_lengths):.1f}"
        )

        # Log amino acid distribution info
        train_aa_presence = [dataset.aa_presence[i] for i in train_idx]
        val_aa_presence = [dataset.aa_presence[i] for i in val_idx]

        for aa in "ACDEFGHIKLMNPQRSTVWY":
            train_count = sum(1 for aa_set in train_aa_presence if aa in aa_set)
            val_count = sum(1 for aa_set in val_aa_presence if aa in aa_set)
            train_pct = 100 * train_count / len(train_idx)
            val_pct = 100 * val_count / len(val_idx)
            if train_pct < 20 or val_pct < 20:  # Log rare AAs
                self.data_logger.info(
                    f"AA '{aa}' - Train: {train_pct:.1f}%, Val: {val_pct:.1f}%"
                )

        self.max_length = max(all_lengths)
        self.min_length = min(all_lengths)
        self.mean_length = sum(all_lengths) / len(all_lengths)
        self.dataset_length = len(dataset)
        self.gap_percentage = dataset.total_gap_count / (
            sum(all_lengths) + dataset.total_gap_count
        )
        self.msa_columns = len(dataset.sequences[0])
        self.num_labels = len(dataset.label_to_idx) if dataset.label_to_idx else 0
        self.label_to_idx = dataset.label_to_idx
        self.idx_to_label = dataset.idx_to_label
        if all_lengths:
            self.p10 = dataset.p10
            self.p25 = dataset.p25
            self.p75 = dataset.p75
            self.p90 = dataset.p90

        if self.num_labels > 0:
            self.data_logger.info(f"Number of labels: {self.num_labels}")
            self.data_logger.info(f"Label names: {self.idx_to_label.values()}")
            self.data_logger.info("Label distribution in train set:")
            train_labels = [
                dataset.label_ids[i] for i in train_idx if dataset.label_ids[i] is not None
            ]
            for label_idx, label_name in self.idx_to_label.items():
                count = sum(1 for c in train_labels if c == label_idx)
                pct = 100 * count / len(train_labels) if train_labels else 0
                self.data_logger.info(f"  {label_name}: {pct:.1f}%")
        # Checked by train.py when label conditioning is on (train and validation both need labels)
        self.num_unlabelled = sum(1 for c in dataset.label_ids if c is None)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )
