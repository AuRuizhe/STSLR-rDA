#!/usr/bin/env python3
# udda_de.py
"""
UDDA (GDD + LSD, dynamic schedule) with 5-band DE features.

Core pipeline:
    1) Raw EEG segments (time-domain)  ->  5-band DE features
    2) Encoder (MLP) extracts feature vectors
    3) Classifier predicts labels
    4) Domain adaptation loss:
           - GDD: global MMD between source and target features
           - LSD: local semantic discrepancy with pseudo labels on target
       with a dynamic schedule between GDD and LSD.

Data assumptions for real usage:
    - Source:
        X_src_raw: (N_src, T, C) time-domain EEG (time × channels)
        y_src:     (N_src,) integer labels in {0, 1}
    - Target:
        X_tgt_raw: (N_tgt, T, C) time-domain EEG
        y_tgt:     (N_tgt,) integer labels in {0, 1} (used for evaluation)

In this file, `main()` uses random synthetic data to demonstrate the
complete training and evaluation flow. To run on real data, simply
replace the synthetic `X_src_raw`, `y_src`, `X_tgt_raw`, `y_tgt` with
your own arrays that follow the same shapes.
"""

import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# Settings
# ============================================================

# Sampling rate and 5 DE bands
FS = 200.0
BANDS = [
    (1, 3),    # delta
    (4, 7),    # theta
    (8, 13),   # alpha
    (14, 30),  # beta
    (31, 35),  # gamma
]

# Training hyperparameters
BATCH_SIZE_SRC = 128
BATCH_SIZE_TGT = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_DA = 0.30
CONF_THRESH = 0.60
KERNEL_NUM = 5
KERNEL_MUL = 2.0
SWITCH_FRAC = 0.60   # fraction of epochs where schedule bends from GDD to LSD
SEED = 42


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device: str):
    """Move tensor or list/tuple of tensors to a device."""
    if isinstance(x, (list, tuple)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


# ============================================================
# 5-band DE feature extraction
# ============================================================

def _band_indices(freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """Return indices of frequencies within [f_lo, f_hi]."""
    return np.where((freqs >= f_lo) & (freqs <= f_hi))[0]


def _de_from_bandpower(P_band: np.ndarray) -> np.ndarray:
    """
    Differential entropy under Gaussian assumption:
        DE = 0.5 * ln(2 * pi * e * P_band)
    """
    eps = 1e-12
    return 0.5 * np.log(2.0 * np.pi * np.e * (P_band + eps))


def de_5bands_batched(
    X_ntc: np.ndarray,
    fs: float = FS,
    bands=BANDS,
    batch: int = 256,
) -> np.ndarray:
    """
    Compute 5-band differential entropy (DE) features from raw EEG.

    Args:
        X_ntc: (N, T, C) array, time-domain signals
        fs: sampling frequency
        bands: list of (f_lo, f_hi)
        batch: batch size for FFT to control memory

    Returns:
        X_de: (N, C * len(bands)) DE features
              bands are concatenated in the order given in `bands`.
    """
    X_ntc = np.asarray(X_ntc)
    N, T, C = X_ntc.shape

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)  # (F,)
    F = freqs.shape[0]
    df = freqs[1] - freqs[0] if F > 1 else (fs / T)

    # precompute band indices
    band_idx = [_band_indices(freqs, lo, hi) for (lo, hi) in bands]

    out = np.empty((N, C * len(bands)), dtype=np.float32)
    for s in range(0, N, batch):
        e = min(N, s + batch)
        x = X_ntc[s:e].astype(np.float32, copy=False)  # (nB, T, C)
        Xf = np.fft.rfft(x, axis=1)                    # (nB, F, C) complex
        Pxx = (np.real(Xf) ** 2 + np.imag(Xf) ** 2) / (T * fs)  # (nB, F, C)

        feats = []
        for idx in band_idx:
            if idx.size == 0:
                P_band = np.zeros((e - s, C), dtype=np.float32)
            else:
                P_band = Pxx[:, idx, :].sum(axis=1) * df        # (nB, C)
            de = _de_from_bandpower(P_band).astype(np.float32)  # (nB, C)
            feats.append(de)

        out[s:e] = np.concatenate(feats, axis=1)                # (nB, C * len(bands))

        del Xf, Pxx
    return out


def standardize_train_apply(
    Xtr: np.ndarray,
    Xte: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score features based on training statistics only.

    Args:
        Xtr: (N_train, D)
        Xte: (N_test, D)

    Returns:
        Xtr_z: standardized training features
        Xte_z: standardized test features
    """
    mu = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-8
    return (Xtr - mu) / std, (Xte - mu) / std


# ============================================================
# Model definitions
# ============================================================

class Encoder(nn.Module):
    """
    Simple MLP encoder for DE features.

    Input:
        x: (B, in_dim)
    Output:
        f: (B, feat_dim)
    """
    def __init__(self, in_dim: int, feat_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier(nn.Module):
    """
    Linear classifier on top of encoder features.
    """
    def __init__(self, in_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f)


# ============================================================
# Kernels & discrepancy measures (GDD & LSD)
# ============================================================

@torch.no_grad()
def _median_sigma2(x: torch.Tensor) -> torch.Tensor:
    """
    Median of squared distances in a random subset, used as bandwidth.
    """
    n = x.size(0)
    k = min(500, n)
    idx = torch.randperm(n, device=x.device)[:k]
    sub = x[idx]
    d2 = torch.cdist(sub, sub, p=2).pow(2)
    d2 = d2[d2 > 0]
    if d2.numel() == 0:
        return torch.tensor(1.0, device=x.device)
    return torch.median(d2)


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float = KERNEL_MUL,
    kernel_num: int = KERNEL_NUM,
    fix_sigma: float = None,
) -> torch.Tensor:
    """
    Multi-kernel Gaussian kernel matrix.

    Args:
        x: (n, D)
        y: (m, D)
        kernel_mul: multiplicative scaling factor for bandwidth
        kernel_num: number of kernels
        fix_sigma: if not None, use this sigma^2 instead of median heuristic

    Returns:
        K: (n + m, n + m) kernel matrix
    """
    total = torch.cat([x, y], dim=0)   # (n + m, D)
    d2 = torch.cdist(total, total, p=2).pow(2)

    if fix_sigma is None:
        sigma2 = _median_sigma2(total)
    else:
        sigma2 = torch.tensor(fix_sigma, device=total.device, dtype=total.dtype)

    K = 0.0
    for i in range(-(kernel_num // 2), (kernel_num // 2) + 1):
        beta = 1.0 / (sigma2 * (kernel_mul ** i))
        K = K + torch.exp(-beta * d2)
    return K


def mmd_unbiased_from_kernel(K: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Unbiased MMD estimate given a precomputed kernel matrix.

    Args:
        K: (n + m, n + m) kernel matrix for [x; y]
        n: number of samples in x
        m: number of samples in y
    """
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]

    if n > 1:
        xx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    else:
        xx = Kxx.mean()

    if m > 1:
        yy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    else:
        yy = Kyy.mean()

    xy = Kxy.mean()
    return xx + yy - 2.0 * xy


def loss_gdd(
    fs: torch.Tensor,
    ft: torch.Tensor,
    kernel_mul: float = KERNEL_MUL,
    kernel_num: int = KERNEL_NUM,
) -> torch.Tensor:
    """
    Global distribution distance (GDD) loss via MMD between source and target features.
    """
    K = gaussian_kernel(fs, ft, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=None)
    return mmd_unbiased_from_kernel(K, fs.size(0), ft.size(0))


def _class_indices(y: torch.Tensor, c: int) -> torch.Tensor:
    """Indices of samples with class label c."""
    return (y == c).nonzero(as_tuple=False).view(-1)


def loss_lsd(
    fs: torch.Tensor,
    ys: torch.Tensor,
    ft: torch.Tensor,
    yt: torch.Tensor,
    kernel_mul: float = KERNEL_MUL,
    kernel_num: int = KERNEL_NUM,
) -> torch.Tensor:
    """
    Local semantic discrepancy (LSD) loss.

    Encourages:
        - small MMD for same-class source/target pairs (intra-class alignment)
        - large MMD for different-class source/target pairs (inter-class separation)

    Args:
        fs: source features (n_s, D)
        ys: source labels   (n_s,)
        ft: target features (n_t, D) (only those with high-confidence pseudo labels)
        yt: pseudo target labels (n_t,)

    Returns:
        Scalar loss value.
    """
    if fs.size(0) == 0 or ft.size(0) == 0 or yt.numel() == 0:
        return torch.tensor(0.0, device=fs.device)

    C = int(torch.max(torch.cat([ys, yt], dim=0)).item()) + 1
    intra_sum, intra_cnt = 0.0, 0
    inter_sum, inter_cnt = 0.0, 0

    # Intra-class alignment
    for c in range(C):
        idx_s = _class_indices(ys, c)
        idx_t = _class_indices(yt, c)
        if idx_s.numel() >= 2 and idx_t.numel() >= 2:
            Ks = gaussian_kernel(fs[idx_s], ft[idx_t], kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=None)
            mmd_c = mmd_unbiased_from_kernel(Ks, idx_s.numel(), idx_t.numel())
            intra_sum += mmd_c
            intra_cnt += 1

    # Inter-class separation
    for c1 in range(C):
        for c2 in range(C):
            if c1 == c2:
                continue
            idx_s = _class_indices(ys, c1)
            idx_t = _class_indices(yt, c2)
            if idx_s.numel() >= 2 and idx_t.numel() >= 2:
                Kdiff = gaussian_kernel(fs[idx_s], ft[idx_t], kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=None)
                mmd_diff = mmd_unbiased_from_kernel(Kdiff, idx_s.numel(), idx_t.numel())
                inter_sum += mmd_diff
                inter_cnt += 1

    if intra_cnt == 0 and inter_cnt == 0:
        return torch.tensor(0.0, device=fs.device)
    if intra_cnt == 0:
        return - inter_sum / max(1, inter_cnt)
    if inter_cnt == 0:
        return intra_sum / max(1, intra_cnt)
    return (intra_sum / max(1, intra_cnt)) - (inter_sum / max(1, inter_cnt))


# ============================================================
# Training & evaluation
# ============================================================

def train_one_domain_adaptation(
    encoder: nn.Module,
    clf: nn.Module,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    epochs: int,
    lr: float,
    wd: float,
    lambda_da: float,
    conf_thresh: float,
    switch_at: int,
    device: str,
) -> None:
    """
    Train UDDA (GDD + LSD with dynamic schedule) on source and target loaders.

    Args:
        encoder, clf: models
        src_loader: labeled source domain loader
        tgt_loader: unlabeled target domain loader
        epochs: number of training epochs
        lr: learning rate
        wd: weight decay
        lambda_da: weight for domain adaptation loss
        conf_thresh: confidence threshold for pseudo labels on target
        switch_at: epoch (around) where schedule bends from GDD to LSD
        device: "cpu" or "cuda"
    """
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=wd)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    ce_loss = nn.CrossEntropyLoss()

    tgt_iter = iter(tgt_loader)

    for ep in range(1, epochs + 1):
        encoder.train()
        clf.train()
        epoch_loss = 0.0
        n_batches = 0

        for xs, ys in src_loader:
            try:
                xt, yt = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, yt = next(tgt_iter)

            xs, ys = to_device(xs, device), to_device(ys, device)
            xt, yt = to_device(xt, device), to_device(yt, device)

            # Source supervised loss
            fs = encoder(xs)
            logit_s = clf(fs)
            loss_ce = ce_loss(logit_s, ys)

            # Target features
            ft = encoder(xt)

            # Dynamic schedule: sigma ∈ (0,1), increases with epoch
            sigma = 1.0 / (1.0 + math.exp(switch_at - ep))

            # Global distribution distance (GDD)
            loss_g = loss_gdd(fs, ft)

            # Local semantic discrepancy (LSD) with pseudo labels
            with torch.no_grad():
                prob_t = F.softmax(clf(ft), dim=1)
                conf, pred = prob_t.max(dim=1)
                mask = conf >= conf_thresh

            if mask.sum().item() >= 2:
                loss_l = loss_lsd(fs, ys, ft[mask], pred[mask])
            else:
                loss_l = torch.tensor(0.0, device=device)

            # Combined loss
            loss = loss_ce + lambda_da * ((1.0 - sigma) * loss_g + sigma * loss_l)

            enc_opt.zero_grad()
            clf_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            clf_opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0 and (ep % 10 == 0 or ep == 1):
            print(f"[Epoch {ep:03d}] loss={epoch_loss / n_batches:.4f} | sigma={sigma:.3f}")


def evaluate(
    encoder: nn.Module,
    clf: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 1024,
) -> float:
    """
    Evaluate classification accuracy on labeled data.

    Args:
        encoder, clf: trained models
        X: features, (N, D)
        y: labels, (N,)
        device: "cpu" or "cuda"
        batch_size: batch size for evaluation

    Returns:
        Accuracy in [0, 1].
    """
    encoder.eval()
    clf.eval()
    acc_num, tot = 0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            xb = to_device(X[i : i + batch_size], device)
            yb = to_device(y[i : i + batch_size], device)
            logits = clf(encoder(xb))
            pred = torch.argmax(logits, dim=1)
            acc_num += (pred == yb).sum().item()
            tot += yb.numel()
    return acc_num / max(1, tot)


# ============================================================
# Main: minimal synthetic example
# ============================================================

def main() -> None:
    """
    Minimal toy demo to verify that UDDA + DE pipeline runs end-to-end.

    Example data shapes:
        B_src : number of source trials
        B_tgt : number of target trials
        T     : number of time points per trial
        C     : number of EEG channels per trial

    In the demo below:
        B_src = 200, B_tgt = 200, T = 256, C = 16

    Replace the synthetic `X_src_raw`, `y_src`, `X_tgt_raw`, `y_tgt`
    with real data in the same shapes to run UDDA on your dataset.
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Fs={FS} Hz | Bands={BANDS}")

    # ------------------------------------------------------------------
    # 1) Synthetic raw data (replace this part with your real data)
    # ------------------------------------------------------------------
    B_src, B_tgt = 200, 200   # number of source / target trials
    T, C = 1000, 62           # time points per trial, number of channels

    rng = np.random.default_rng(SEED)
    X_src_raw = rng.standard_normal((B_src, T, C))  # (B_src, T, C)
    X_tgt_raw = rng.standard_normal((B_tgt, T, C))  # (B_tgt, T, C)
    y_src = rng.integers(0, 2, size=B_src, dtype=np.int64)
    y_tgt = rng.integers(0, 2, size=B_tgt, dtype=np.int64)

    print(
        f"Source raw: {X_src_raw.shape}  (B_src={B_src}, T={T}, C={C})\n"
        f"Target raw: {X_tgt_raw.shape}  (B_tgt={B_tgt}, T={T}, C={C})"
    )

    # ------------------------------------------------------------------
    # 2) 5-band DE features
    # ------------------------------------------------------------------
    t_feat = time.time()
    X_src_de = de_5bands_batched(X_src_raw, fs=FS, bands=BANDS, batch=128)
    X_tgt_de = de_5bands_batched(X_tgt_raw, fs=FS, bands=BANDS, batch=128)
    print(
        f"DE features: Source {X_src_de.shape} | Target {X_tgt_de.shape} "
        f"| time {time.time() - t_feat:.2f}s"
    )

    # ------------------------------------------------------------------
    # 3) Standardization (train stats only)
    # ------------------------------------------------------------------
    X_src_z, X_tgt_z = standardize_train_apply(X_src_de, X_tgt_de)
    in_dim = X_src_z.shape[1]
    print(
        f"Feature dim = {in_dim} (= C * {len(BANDS)}) | "
        f"N_src={X_src_z.shape[0]} | N_tgt={X_tgt_z.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 4) DataLoaders
    # ------------------------------------------------------------------
    Xs = torch.from_numpy(X_src_z).float()
    ys = torch.from_numpy(y_src).long()
    Xt = torch.from_numpy(X_tgt_z).float()
    yt = torch.from_numpy(y_tgt).long()

    bs_src = min(BATCH_SIZE_SRC, Xs.size(0))
    bs_tgt = min(BATCH_SIZE_TGT, Xt.size(0))

    src_loader = DataLoader(
        TensorDataset(Xs, ys),
        batch_size=bs_src,
        shuffle=True,
        drop_last=True,
    )
    tgt_loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=bs_tgt,
        shuffle=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 5) Models
    # ------------------------------------------------------------------
    encoder = Encoder(in_dim=in_dim, feat_dim=64, dropout=0.2).to(device)
    clf = Classifier(in_dim=64, num_classes=2).to(device)

    # ------------------------------------------------------------------
    # 6) Train UDDA
    # ------------------------------------------------------------------
    print("\nTraining UDDA on synthetic source/target domains...")
    t0 = time.time()
    switch_at = int(SWITCH_FRAC * EPOCHS)
    train_one_domain_adaptation(
        encoder,
        clf,
        src_loader,
        tgt_loader,
        epochs=EPOCHS,
        lr=LR,
        wd=WEIGHT_DECAY,
        lambda_da=LAMBDA_DA,
        conf_thresh=CONF_THRESH,
        switch_at=switch_at,
        device=device,
    )
    dt = time.time() - t0

    # ------------------------------------------------------------------
    # 7) Evaluate on target domain (using true labels)
    # ------------------------------------------------------------------
    acc = evaluate(encoder, clf, Xt, yt, device=device)
    print(
        f"\n=== UDDA demo finished ===\n"
        f"Target accuracy: {acc * 100:.2f}% | Train time: {dt:.2f}s"
    )


if __name__ == "__main__":
    main()
