#!/usr/bin/env python3
# dan_offline.py
"""
Offline DAN (GRU + MMD) example without any online/streaming adaptation.

- Model structure:
    * GRU(input_size=5, hidden_size=16)
    * Fully-connected layer: 62*16 -> 512 (BN + ReLU)
    * Classification head: 512 -> 3 classes
    * MMD with multi-kernel Gaussian kernel

- Data:
    This script uses random toy data of shape (N, 62, 5), mimicking
    62 EEG channels × 5 DE bands. For real experiments, replace
    `generate_toy_data()` with your own data-loading function that
    returns DE features and labels.

Run:
    python dan_offline.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ====================== 1. Dataset wrapper ======================

class EEGDataset(Dataset):
    """
    Simple EEG DE feature dataset.

    x: (N, 62, 5) or (N, n_channels, n_bands)
    y: (N,) integer labels, or None for unlabeled target data.
    """
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        if y is None:
            self.y = None
        else:
            self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        if self.y is None:
            # Unlabeled target domain: return a dummy label
            return self.x[idx], torch.tensor(0, dtype=torch.long)
        return self.x[idx], self.y[idx]


# ====================== 2. MMD utilities ======================

def gaussian_kernel(
    src: torch.Tensor,
    tgt: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """
    Multi-kernel Gaussian kernel matrix used in MMD.

    Args:
        src: (B_s, D)
        tgt: (B_t, D)
        kernel_mul: bandwidth multiplier.
        kernel_num: number of kernels.

    Returns:
        K: (B_s + B_t, B_s + B_t) kernel matrix.
    """
    total = torch.cat([src, tgt], dim=0)  # (N_total, D)
    L2 = ((total[:, None, :] - total[None, :, :]) ** 2).sum(dim=2)

    n = total.size(0)
    bandwidth = L2.sum() / (n * n - n + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    kernels = [
        torch.exp(-L2 / (bandwidth * (kernel_mul ** i)))
        for i in range(kernel_num)
    ]
    return sum(kernels)


def mmd(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) loss.

    Args:
        src: (B_s, D)
        tgt: (B_t, D)

    Returns:
        Scalar tensor representing the MMD loss.
    """
    n_src = src.size(0)
    n_tgt = tgt.size(0)
    K = gaussian_kernel(src, tgt)
    XX = K[:n_src, :n_src]
    YY = K[n_src:, n_src:]
    XY = K[:n_src, n_src:]
    YX = K[n_src:, :n_src]
    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
    return loss


# ====================== 3. GRU-based DAN model ======================

class GRUModel(nn.Module):
    """
    GRU-based feature extractor + classifier for DAN.

    Input:
        x: (B, 62, 5)  # 62 channels × 5 DE bands

    Output:
        logits: (B, n_classes)
        feat:   (B, feature_dim)  # used for MMD
    """
    def __init__(
        self,
        n_bands: int = 5,
        n_channels: int = 62,
        hidden_size: int = 16,
        feature_dim: int = 512,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels

        # GRU: input_size = number of DE bands, hidden_size = 16
        self.gru = nn.GRU(
            input_size=n_bands,
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Flatten all time steps (channels) and map to feature_dim
        self.feat = nn.Sequential(
            nn.Linear(n_channels * hidden_size, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # Classification head
        self.cls = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_channels, n_bands) = (B, 62, 5)

        Returns:
            logits: (B, n_classes)
            feat:   (B, feature_dim)
        """
        # GRU expects (batch, seq_len, input_size) = (B, 62, 5)
        h, _ = self.gru(x)                         # (B, 62, hidden_size)
        feat = self.feat(h.reshape(x.size(0), -1)) # (B, feature_dim)
        logits = self.cls(feat)                    # (B, n_classes)
        return logits, feat


# ====================== 4. Toy data generation ======================

def generate_toy_data(
    n_src: int = 600,
    n_tgt_unl: int = 600,
    n_tgt_test: int = 300,
    n_channels: int = 62,
    n_bands: int = 5,
    n_classes: int = 2,
    seed: int = 0,
):
    """
    Generate random DE-like features for a toy domain adaptation example.

    In real experiments, replace this function with your own loader
    that returns:
        X_src, y_src, X_tgt_u, X_tgt_te, y_tgt_te

    Returns:
        X_src:    (n_src, n_channels, n_bands)
        y_src:    (n_src,)
        X_tgt_u:  (n_tgt_unl, n_channels, n_bands)   # unlabeled target domain
        X_tgt_te: (n_tgt_test, n_channels, n_bands)  # target test set
        y_tgt_te: (n_tgt_test,)
    """
    g = torch.Generator().manual_seed(seed)

    X_src = torch.randn(n_src, n_channels, n_bands, generator=g)
    X_tgt_u = torch.randn(n_tgt_unl, n_channels, n_bands, generator=g)
    X_tgt_te = torch.randn(n_tgt_test, n_channels, n_bands, generator=g)

    y_src = torch.randint(0, n_classes, (n_src,), generator=g)
    y_tgt_te = torch.randint(0, n_classes, (n_tgt_test,), generator=g)

    return (
        X_src.numpy(),
        y_src.numpy(),
        X_tgt_u.numpy(),
        X_tgt_te.numpy(),
        y_tgt_te.numpy(),
    )


# ====================== 5. Evaluation on labeled data ======================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate the model on labeled data.

    Returns:
        avg_loss: average cross-entropy loss
        acc:      classification accuracy
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (preds == y).sum().item()

    return total_loss / total, correct / total


# ====================== 6. Offline UDA training loop ======================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------- 1) Generate toy data (replace with your real data loader) --------
    (
        X_src,      # labeled source domain
        y_src,
        X_tgt_u,    # unlabeled target domain for MMD
        X_tgt_te,   # labeled target test set
        y_tgt_te,
    ) = generate_toy_data()

    src_dataset = EEGDataset(X_src, y_src)
    tgt_unl_dataset = EEGDataset(X_tgt_u, None)
    tgt_test_dataset = EEGDataset(X_tgt_te, y_tgt_te)

    src_loader = DataLoader(src_dataset, batch_size=128, shuffle=True)
    tgt_unl_loader = DataLoader(tgt_unl_dataset, batch_size=128, shuffle=True)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=256, shuffle=False)

    # -------- 2) Build model / loss / optimizer --------
    model = GRUModel(n_bands=5, n_channels=62, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lam = 0.5  # MMD weight

    # -------- 3) Offline UDA training --------
    n_epochs = 5000
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_mmd = 0.0
        n_steps = 0

        tgt_iter = iter(tgt_unl_loader)

        for x_s, y_s in src_loader:
            # fetch one target batch; restart iterator if exhausted
            try:
                x_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_unl_loader)
                x_t, _ = next(tgt_iter)

            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            logits_s, feat_s = model(x_s)
            _, feat_t = model(x_t)

            cls_loss = criterion(logits_s, y_s)
            mmd_loss_val = mmd(feat_s, feat_t)
            loss = cls_loss + lam * mmd_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_mmd += mmd_loss_val.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        avg_cls = total_cls / n_steps
        avg_mmd = total_mmd / n_steps

        # Evaluate on labeled target test set
        test_loss, test_acc = evaluate(model, tgt_test_loader, criterion, device)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Train loss: {avg_loss:.4f} "
                f"(cls {avg_cls:.4f}, mmd {avg_mmd:.4f}) | "
                f"Target test: acc {test_acc*100:.1f}%, "
                f"loss {test_loss:.4f}"
            )


if __name__ == "__main__":
    main()
