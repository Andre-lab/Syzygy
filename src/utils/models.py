import math

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.special import gammaln


# Re-implementation of the Dirichlet flow model from the original research repo
# This implementation calculates the majority on the GPU for large speedups
# With some CPU precomputation for the beta CDFs and their derivatives
class DirichletConditionalFlowHybrid:
    def __init__(
        self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01, device="cuda"
    ):
        self.K = K
        self.device = device
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_spacing = alpha_spacing

        # Precompute on CPU first
        alphas_np = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        bs_np = np.linspace(0, 1, 1000)

        # Compute beta CDFs (CPU-bound, but only once)
        beta_cdfs = []
        for alph in alphas_np:
            beta_cdfs.append(scipy.special.betainc(alph, K - 1, bs_np))
        beta_cdfs_np = np.array(beta_cdfs)

        # Compute derivatives
        beta_cdfs_derivative_np = np.diff(beta_cdfs_np, axis=0) / alpha_spacing

        # Convert to PyTorch tensors
        self.alphas_tensor = torch.tensor(
            alphas_np, dtype=torch.float32, device=self.device
        )
        self.bs_tensor = torch.tensor(bs_np, dtype=torch.float32, device=self.device)
        self.beta_cdfs_derivative = torch.tensor(
            beta_cdfs_derivative_np, dtype=torch.float32, device=self.device
        )

        # Precompute log beta values
        self.log_beta_const = gammaln(
            torch.tensor(K - 1, dtype=torch.float32, device=self.device)
        )
        self.log_beta_values = (
            gammaln(self.alphas_tensor)
            + self.log_beta_const
            - gammaln(self.alphas_tensor + K - 1)
        )

        # Precompute interpolation indices (fixed grid)
        self.grid_size = len(bs_np)
        self.grid_indices = torch.arange(0, len(bs_np), device=self.device)

    def c_factor(self, bs, alpha):
        # Ensure tensors on correct device
        bs = bs.float()
        alpha = (
            alpha.float()
            if isinstance(alpha, torch.Tensor)
            else torch.tensor(alpha, device=self.device).float()
        )

        # Find closest alpha index (vectorized)
        alpha_idx = torch.abs(self.alphas_tensor - alpha).argmin()
        alpha_idx = torch.clamp(alpha_idx, 0, len(self.alphas_tensor) - 2)

        # Get derivative function
        I_func = self.beta_cdfs_derivative[alpha_idx]

        # Fast grid-based interpolation
        positions = bs * (self.grid_size - 1)
        indices = positions.long()
        weights = positions - indices.float()

        # Handle boundary conditions
        indices = torch.clamp(indices, 0, self.grid_size - 2)
        next_indices = indices + 1

        # Gather values
        y0 = I_func[indices]
        y1 = I_func[next_indices]
        interp = -(y0 + weights * (y1 - y0))

        # Compute beta component
        log_beta_val = self.log_beta_values[alpha_idx]
        out1 = torch.exp(log_beta_val)

        # Vectorized conditional computation
        safe_bs = torch.clamp(bs, 1e-10, 1 - 1e-10)
        out2 = torch.where(bs < 1, out1 / (1 - safe_bs) ** (self.K - 1), 0.0)

        # Handle bs=0 cases
        exponent = alpha - 1
        term = torch.where(
            (bs == 0) & (exponent == 0),
            torch.ones_like(bs),
            torch.pow(safe_bs, exponent),
        )

        # Final output
        out = torch.where(term > 0, out2 / term, 0.0)
        result = interp * out

        return torch.nan_to_num(result, 0.0)

    def to(self, device):
        self.device = device
        self.alphas_tensor = self.alphas_tensor.to(device)
        self.bs_tensor = self.bs_tensor.to(device)
        self.beta_cdfs_derivative = self.beta_cdfs_derivative.to(device)
        self.log_beta_values = self.log_beta_values.to(device)
        return self


def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[
        :, None, None
    ]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights


def simplex_proj(seq: torch.Tensor) -> torch.Tensor:
    """
    Algorithm from Weiran Wang & Carreira-Perpiñán (2013)
    Projects each row of `seq` onto the probability simplex.
    """
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization for time conditioning"""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # AdaLN modulation - learns scale and shift from time/condition embeddings
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: [B, L, D] sequence features
            cond: [B, D] conditioning (time + length embeddings)
        """
        B, L, D = x.shape

        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * h

        # Feed-forward with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        h = self.ffn(h)
        x = x + gate_ffn.unsqueeze(1) * h

        return x


class AdaLNTransformerModel(nn.Module):
    def __init__(
        self,
        cfg,
        alphabet_size,
        num_cls,
        use_length_conditioning=True,
        use_label_conditioning=False,
        num_labels: int | None = None,
        min_ungapped_length=50,
        max_ungapped_length=500,
        max_seq_len=512,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.num_cls = num_cls
        self.cfg = cfg
        self.use_length_conditioning = use_length_conditioning
        self.use_label_conditioning = use_label_conditioning
        self.min_ungapped_length = min_ungapped_length
        self.max_ungapped_length = max_ungapped_length
        self.max_seq_len = max_seq_len

        # Token embedder
        self.embedder = nn.Linear(2 * self.alphabet_size, cfg.hidden_dim)

        # Time embedder
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # Length embedder
        if self.use_length_conditioning:
            self.length_embedding = nn.Sequential(
                nn.Linear(1, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )

        # Label embedder (discrete labels)
        if self.use_label_conditioning:
            if not num_labels or num_labels <= 0:
                raise ValueError(
                    "`use_label_conditioning=True` but `num_labels` was not provided "
                    "or is <= 0. Pass the number of unique labels in the dataset."
                )
            self.num_labels = int(num_labels)
            self.label_embedding = nn.Embedding(self.num_labels, cfg.hidden_dim)

        # Fuse conditioning sources into a single hidden_dim vector for AdaLN.
        # We deliberately *do not* just sum embeddings: fusing via an MLP lets the model
        # learn richer interactions between time/length/label.
        cond_parts = (
            1 + int(self.use_length_conditioning) + int(self.use_label_conditioning)
        )
        if cond_parts > 1:
            self.cond_fuser = nn.Sequential(
                nn.SiLU(), nn.Linear(cond_parts * cfg.hidden_dim, cfg.hidden_dim)
            )
        else:
            self.cond_fuser = None

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    dim_feedforward=4 * cfg.hidden_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim)
        )
        self.out = nn.Linear(cfg.hidden_dim, self.alphabet_size)

        # Initialize output layer to near-zero (helps training stability)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def sinusoidal_pe(self, L, device):
        """Sinusoidal positional encoding"""
        pe = torch.zeros(L, self.cfg.hidden_dim, device=device)
        position = torch.arange(0, L, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.cfg.hidden_dim, 2, device=device)
            * -(math.log(10000.0) / self.cfg.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, seq, t, length=None, label=None):
        B, L, K = seq.shape

        # Embed sequence
        feat = self.embedder(seq)  # [B, L, hidden_dim]

        # Add positional encoding
        pe = self.sinusoidal_pe(L, seq.device)
        feat = feat + pe.unsqueeze(0)

        # Create conditioning vector (time + optional length + optional label)
        time_cond = self.time_embedder(t)  # [B, hidden_dim]
        cond_parts = [time_cond]

        if self.use_length_conditioning:
            if length is None:
                raise ValueError(
                    "`use_length_conditioning=True` but `length` was not provided."
                )
            norm_length = (length.float() - self.min_ungapped_length) / (
                self.max_ungapped_length - self.min_ungapped_length
            )
            norm_length = torch.tanh(norm_length).unsqueeze(-1)
            length_emb = self.length_embedding(norm_length)
            cond_parts.append(length_emb)

        if self.use_label_conditioning:
            if label is None:
                raise ValueError(
                    "`use_label_conditioning=True` but `label` was not provided."
                )
            if label.dtype != torch.long:
                label = label.long()
            if (label < 0).any() or (label >= self.num_labels).any():
                raise ValueError(
                    "Invalid `label` indices passed to forward(). "
                    "Must be in range [0, num_labels)."
                )
            label_emb = self.label_embedding(label)
            cond_parts.append(label_emb)

        if self.cond_fuser is not None:
            cond = self.cond_fuser(torch.cat(cond_parts, dim=-1))
        else:
            cond = cond_parts[0]

        # Apply transformer blocks with conditioning
        for block in self.blocks:
            feat = block(feat, cond)

        # Final layer with AdaLN
        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        feat = self.final_norm(feat)
        feat = feat * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.out(feat)
