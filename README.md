# Sparse Spatio-Temporal Logistic Regression with Regularized Domain Alignment (STSLR-rDA)

This repository contains the Python implementation of **STSLR-rDA** and the **baseline methods** used for comparison in our experiments on EEG-based emotion recognition.

> **Note:** This code is provided for **reproducibility and inspection**.  
> The EEG datasets used in the paper are **not included** due to license and size constraints (see [Data](#data) below).

---

## 1. STSLR-rDA (Proposed Method)

STSLR-rDA is a sparse spatio-temporal logistic regression framework with **regularized domain alignment** in the log-SPD space of covariance matrices.  

Key steps:

1. **Time-delay concatenation** of EEG time series to capture short-range temporal dependencies.
2. **Trace-normalized covariance** computation per trial.
3. **Matrix logarithm** to map SPD covariance matrices to a Euclidean log-SPD space.
4. **Regularized CORAL-based domain alignment** (source → target) in the log-SPD feature space.
5. **Sparse logistic regression** with combined **L1** and **L21** regularization (row-wise group sparsity) trained via an **accelerated proximal gradient method (APGM)**.

### Main files

- `STSLR-rDA_main.py`  
  Minimal entry point showing how to:
  - load or simulate data with shape `(N, T, C)`  
    - `N`: number of trials  
    - `T`: number of time samples per trial  
    - `C`: number of EEG channels  
  - call the domain alignment pipeline  
  - train the STSLR model and evaluate test accuracy.

- `domain_align.py`  
  Core domain-alignment and feature extraction components:
  - `DelayConcat`: time-delay concatenation along the channel dimension.  
  - `CovTraceNoLog`: per-trial covariance with trace normalization.  
  - `LogmTransformer`: matrix logarithm of SPD covariance matrices.  
  - `GlobalCORAL_LogSPD`: global regularized CORAL transform in log-SPD space (source → target).  
  - `domain_align(...)`: high-level function  
    ```python
    Rs_train, Rs_test = domain_align(X_train, X_test, tau_delay=1, ...)
    ```
    where `X_train` and `X_test` are raw time-series arrays of shape `(N, T, C)`,  
    and `Rs_train`, `Rs_test` are log-SPD matrices of shape `(N, M, M)` used by the model.

- `log_L21_L1.py`  
  Sparse spatio-temporal logistic regression:
  - `train_LogL21_L1(...)`: APGM solver for logistic regression with combined L1 + L21 regularization on a weight matrix `W ∈ ℝ^{M×M}`.
  - `Class_L21L1`: convenient wrapper with a scikit-learn–like interface:
    ```python
    clf = Class_L21L1(lam_l1=..., lam_l21=...)
    clf.fit(Rs_train, y_train)
    acc = clf.score(Rs_test, y_test)
    ```

---

## 2. Baseline Methods

The repository also includes implementations of the **baseline methods** used for comparison with STSLR-rDA:

- **DAN** – Deep Adaptation Network with MMD-based domain alignment.
- **DGCNN** – Dynamic Graph Convolutional Neural Network operating on EEG time series.
- **UDDA** – Unsupervised DDA for EEG emotion recognition, focusing on both global and local domain discrepancies to enhance feature discriminatingly.
- **GRU-MCC** – GRU-based encoder with **Minimum Class Confusion (MCC)** loss for domain adaptation.
- **SBLECA** – Sparse Bayesian learning on log-SPD covariance features with CORAL-based alignment.

These baselines are organized under:

```text
baselines/
  DAN.py
  DGCNN.py
  UDDA.py
  GRU-MCC.py
  SBLECA.py
