#!/usr/bin/env python3
# gru_mcc.py
"""
GRU-MCC domain adaptation for EEG with 5-band DE features.

Core pipeline:
    1) Raw EEG segments (C x T) -> 5-band DE features (C x 5)
    2) Stack over trials: (N, C, 5)
    3) GRU over the channel dimension (sequence length = C, input dim = 5)
    4) Supervised CE loss on source domain + MCC loss on target domain

Expected real data shapes:
    - X_src_raw: (N_src, C, T)
    - y_src    : (N_src,) labels in {0, 1} or {-1, 1}
    - X_tgt_raw: (N_tgt, C, T)
    - y_tgt    : (N_tgt,) labels in {0, 1} or {-1, 1} (for evaluation)

In the __main__ part we show a minimal synthetic example:
    B_src, B_tgt = 128, 128   # number of source / target trials
    C, T = 62, 1000           # channels and time points per trial
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import mne  # only used for band-pass filtering

# ---------------- 1. 5-band DE features -----------------
SFREQ = 200  # sampling rate (Hz)
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # delta, theta, alpha, beta, gamma


def raw2de(sample_ct: np.ndarray) -> np.ndarray:
    """
    Convert one raw EEG segment to 5-band DE features.

    Args:
        sample_ct: (C, T) array, channels x time

    Returns:
        de_feats: (C, 5) float32 DE features, one column per band
    """
    # MNE expects float64
    x = sample_ct.astype("float64", copy=False)

    feats = []
    for lo, hi in BANDS:
        bp = mne.filter.filter_data(
            x, SFREQ, l_freq=lo, h_freq=hi, verbose=False
        )  # (C, T) float64 band-passed data

        # Differential entropy (Gaussian assumption):
        #   0.5 * log(2 * pi * e * var)
        # variance along time axis (axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * np.var(bp, axis=1, ddof=1))
        feats.append(de.astype("float32"))

    # (C, 5) float32
    return np.stack(feats, axis=1)


def convert_dataset(raw_nct: np.ndarray) -> np.ndarray:
    """
    Apply raw2de to a dataset and standardize across trials.

    Args:
        raw_nct: (N, C, T) raw EEG

    Returns:
        out: (N, C, 5) DE features, z-scored over trials for each (C, band)
    """
    out = np.stack([raw2de(s) for s in raw_nct])  # (N, C, 5)
    out = out.astype("float32", copy=False)

    # Z-score across trials for each channel × band
    out -= out.mean(0, keepdims=True)
    out /= out.std(0, keepdims=True) + 1e-8
    return out


# ---------------- 2. Dataset wrapper ----------------------
class EEGset(Dataset):
    """
    Simple dataset:
        x: (N, C, 5)
        y: (N,) or None (for unlabeled target data)
    """

    def __init__(self, x: np.ndarray, y=None):
        self.x = torch.tensor(x)  # (N, C, 5)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i: int):
        if self.y is None:
            return self.x[i], torch.tensor(0)  # dummy label for unlabeled target
        else:
            return self.x[i], self.y[i]


# ---------------- 3. GRU-MCC components --------------------
def entropy(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Entropy of a categorical distribution for each row in p.

    Args:
        p: (B, K) probability distribution over classes

    Returns:
        ent: (B,) entropy values
    """
    return -(p * torch.log(p + eps)).sum(1)


def mcc_loss(logits: torch.Tensor, T: float = 5.0) -> torch.Tensor:
    """
    Minimum Class Confusion (MCC) loss for target logits.

    Args:
        logits: (B, K) logits on target domain
        T: temperature for softmax

    Returns:
        Scalar MCC loss.
    """
    p = torch.softmax(logits / T, dim=1)  # (B, K)
    w = 1 + torch.exp(-entropy(p))        # (B,)
    w = logits.size(0) * w / w.sum()      # re-weighted sample importance

    # Weighted covariance of predictions
    cov = p.mul(w.unsqueeze(1)).t() @ p   # (K, K)
    cov = cov / cov.sum(1, keepdim=True)

    # Sum of off-diagonal entries
    return (cov.sum() - torch.trace(cov)) / logits.size(1)


class GRUModel(nn.Module):
    """
    GRU-based feature extractor + classifier.

    Input x is expected to be (B, C, 5), where:
        B: batch size (number of trials)
        C: number of channels (sequence length)
        5: number of DE bands per channel

    GRU is applied along the channel dimension.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=16, batch_first=True)
        self.feat = nn.Sequential(
            nn.Linear(62 * 16, 512),  # assumes C = 62
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.cls = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, 5)

        Returns:
            logits: (B, 2)
            f:      (B, 512) feature representation
        """
        h, _ = self.gru(x)                      # (B, C, 16)
        f = self.feat(h.reshape(x.size(0), -1)) # (B, 512)
        return self.cls(f), f


# ---------------- 4. Training & evaluation -----------------
def train(
    model: nn.Module,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    epochs: int = 20,
    lam_max: float = 1.0,
    device: str = "cpu",
):
    """
    Joint training on source (supervised CE) and target (unsupervised MCC).

    Args:
        model: GRUModel
        src_loader: labeled source domain loader
        tgt_loader: unlabeled target domain loader
        epochs: number of epochs
        lam_max: maximum weight for MCC loss (linearly scheduled)
        device: "cpu" or "cuda"
    """
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        lam = lam_max * ep / epochs

        ce_running = 0.0
        mcc_running = 0.0
        n_batches = 0

        # zip will stop at the shorter loader; for a more
        # advanced usage you can cycle the longer one instead.
        for (xs, ys), (xt, _) in zip(src_loader, tgt_loader):
            xs = xs.to(device)
            ys = ys.to(device)
            xt = xt.to(device)

            opt.zero_grad()

            ls, _ = model(xs)   # supervised on source
            lt, _ = model(xt)   # unsupervised on target

            loss_ce = ce(ls, ys)
            loss_mcc = mcc_loss(lt)
            loss = loss_ce + lam * loss_mcc

            loss.backward()
            opt.step()

            ce_running += loss_ce.item()
            mcc_running += loss_mcc.item()
            n_batches += 1

        if n_batches == 0:
            print(
                f"E{ep:02d}  (no batches this epoch; "
                f"check batch_size / drop_last settings)"
            )
        else:
            print(
                f"E{ep:02d}  CE={ce_running / n_batches:.3f}  "
                f"MCC_w={lam:.2f}  MCC_raw={mcc_running / n_batches:.3f}"
            )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """
    Evaluate classification accuracy on a labeled dataset.

    Args:
        model: trained GRUModel
        loader: DataLoader that yields (x, y)
        device: "cpu" or "cuda"

    Returns:
        Accuracy in [0, 1].
    """
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        logits, _ = model(x.to(device))
        preds += logits.argmax(1).cpu().tolist()
        gts += y.tolist()

    acc = accuracy_score(gts, preds)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(gts, preds))
    return acc


# ---------------- 5. Minimal synthetic demo ----------------
if __name__ == "__main__":
    """
    Minimal example to verify GRU-MCC runs end-to-end.

    Example data shapes:
        B_src: number of source trials (e.g., 128)
        B_tgt: number of target trials (e.g., 128)
        C:     number of EEG channels (e.g., 62)
        T:     number of time points per trial (e.g., 1000)

    In a real experiment:
        - Replace the synthetic X_src_raw / X_tgt_raw / y_src / y_tgt
          with your own arrays of shape (N, C, T) and labels.
        - If labels are in {-1, 1}, map -1 -> 0 as below.
    """
    # Example dimensions
    B_src, B_tgt = 128, 128   # source / target trial counts
    C, T = 62, 1000           # channels, time points

    rng = np.random.default_rng(42)
    X_src_raw = rng.standard_normal((B_src, C, T)).astype("float32")  # (B_src, C, T)
    X_tgt_raw = rng.standard_normal((B_tgt, C, T)).astype("float32")  # (B_tgt, C, T)

    # Example labels in {0, 1}; if you have {-1, 1}, do y[y == -1] = 0
    y_src = rng.integers(0, 2, size=B_src, dtype=np.int64)
    y_tgt = rng.integers(0, 2, size=B_tgt, dtype=np.int64)

    print(
        f"Source raw: {X_src_raw.shape}  (B_src={B_src}, C={C}, T={T})\n"
        f"Target raw: {X_tgt_raw.shape}  (B_tgt={B_tgt}, C={C}, T={T})"
    )

    # Convert to 5-band DE features and standardize over trials
    train_x = convert_dataset(X_src_raw)  # (B_src, C, 5)
    test_x = convert_dataset(X_tgt_raw)   # (B_tgt, C, 5)

    # DataLoaders
    src_loader = DataLoader(
        EEGset(train_x, y_src), batch_size=64, shuffle=True, drop_last=False
    )
    tgt_loader = DataLoader(
        EEGset(test_x), batch_size=64, shuffle=True, drop_last=False
    )
    tst_loader = DataLoader(
        EEGset(test_x, y_tgt), batch_size=128, shuffle=False, drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    net = GRUModel().to(device)

    # Train and evaluate
    train(net, src_loader, tgt_loader, epochs=20, device=device)
    acc = evaluate(net, tst_loader, device=device)
    print(f"Final accuracy (synthetic demo): {acc * 100:.2f}%")
