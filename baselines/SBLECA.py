#!/usr/bin/env python3
"""
Core implementation of SBLEST with CORAL alignment.

This module provides:
    - vecF: column-major vectorization
    - standardize_labels: map binary labels to {-1, +1}
    - calculate_log_det: numerically-stable logdet
    - logm_spd_*: matrix logarithm for SPD matrices (SciPy / eig / torch)
    - LogmOptions, compute_logm_batch: batch SPD logm with optional parallelism
    - ntc_to_list_ct, _stack_delays: temporal stacking and format conversion
    - compute_covariance_train / compute_covariance_test_nowhiten:
        trace-normalized covariance + logm features
    - SBLESTParams, SBLEST_CORAL_nowhiten:
        SBLEST classifier with CORAL alignment in log-SPD space

No file I/O, dataset-specific code, or result saving is included here.
The caller is responsible for:
    - loading raw data into the required formats
    - converting N×T×C arrays into List[C×T]
    - running cross-subject loops
    - saving accuracies or features

All shapes are written in NumPy / MATLAB-style:
    - C: number of channels
    - T: number of time samples
    - K: number of temporal delays
    - KC: K * C
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------

def vecF(A: np.ndarray) -> np.ndarray:
    """
    Column-major vectorization (MATLAB-style vec).

    Args:
        A: 2D array of shape (m, n)

    Returns:
        1D array of length m * n, flattened in column-major order.
    """
    return np.asarray(A, order="F").reshape(-1, order="F")


def standardize_labels(y_in: np.ndarray) -> np.ndarray:
    """
    Map arbitrary binary labels to {-1, +1}.

    Args:
        y_in: array-like of shape (N,)

    Returns:
        y: float array of shape (N,) with values in {-1.0, +1.0}

    Raises:
        AssertionError: if the number of unique classes is not 2.
    """
    y = np.asarray(y_in, dtype=float).ravel()
    unique_vals = np.unique(y)
    assert unique_vals.size == 2, f"Only binary classification is supported, got {unique_vals.size} classes."
    y[y == unique_vals[0]] = -1.0
    y[y == unique_vals[1]] = +1.0
    return y


def calculate_log_det(X: np.ndarray) -> float:
    """
    Numerically-stable logdet computation, consistent with the original MATLAB implementation.

    Steps:
        1) Scale X by the magnitude of the (0,0) element.
        2) Symmetrize the matrix.
        3) Compute Cholesky factorization and accumulate logdiag(L).

    Args:
        X: SPD matrix of shape (n, n)

    Returns:
        logdet(X) as a float.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    c = 10.0 ** math.floor(math.log10(abs(X[0, 0]) + np.finfo(X.dtype).eps))
    A = X / c
    # symmetrize for numerical stability
    A = (A + A.T) * 0.5

    import scipy.linalg as la

    L = la.cholesky(A, lower=True, overwrite_a=False, check_finite=True)
    log_det_A = 2.0 * np.log(np.diag(L)).sum()
    return n * math.log(c) + log_det_A


# -------------------------------------------------------------------------
# SPD logm utilities
# -------------------------------------------------------------------------

def logm_spd_scipy(C: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm for SPD matrices using SciPy's logm.

    This is numerically very close to MATLAB's logm for SPD matrices.

    Args:
        C: SPD matrix of shape (n, n)

    Returns:
        log(C) of shape (n, n), real part only.
    """
    import scipy.linalg as la

    L = la.logm(C)
    return L.real


def logm_spd_eig(C: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Matrix logarithm for SPD matrices via eigen-decomposition.

    Implements log(C) = Q log(D) Q^T with eigen-decomposition C = Q D Q^T.

    Args:
        C: SPD matrix of shape (n, n)
        eps: minimum eigenvalue threshold

    Returns:
        log(C) of shape (n, n)

    Note:
        This may introduce very small numerical differences (~1e-12) compared
        to SciPy/MATLAB logm.
    """
    C = 0.5 * (C + C.T)
    w, Q = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return (Q * np.log(w)) @ Q.T


def logm_spd_torch(C: np.ndarray, eps: float = 1e-15, device: str = "cuda") -> np.ndarray:
    """
    Matrix logarithm for SPD matrices via eigen-decomposition using PyTorch.

    Args:
        C: SPD matrix of shape (n, n)
        eps: minimum eigenvalue threshold
        device: device for torch.tensor (e.g., "cuda" or "cpu")

    Returns:
        log(C) as a NumPy array of shape (n, n)

    Note:
        Suitable when GPU acceleration is desired. May introduce tiny numerical
        differences compared to SciPy/MATLAB logm.
    """
    import torch

    Ct = torch.tensor(C, dtype=torch.float64, device=device)
    Ct = 0.5 * (Ct + Ct.T)
    w, Q = torch.linalg.eigh(Ct)
    w = torch.clamp(w, min=eps)
    L = (Q * torch.log(w)) @ Q.T
    return L.detach().cpu().numpy()


@dataclass
class LogmOptions:
    """
    Configuration for SPD logm computation.

    Attributes:
        mode: "scipy" (default, closest to MATLAB),
              "eig"   (NumPy eigen-decomposition),
              "torch" (PyTorch eigen-decomposition).
        torch_device: device string for torch (e.g., "cuda").
        eig_eps: minimum eigenvalue threshold in eig-based methods.
        n_jobs: if >1, use joblib.Parallel for batch logm; 0/1 disables parallelism.
    """
    mode: str = "scipy"
    torch_device: str = "cuda"
    eig_eps: float = 1e-15
    n_jobs: int = 0


def compute_logm_batch(covs: List[np.ndarray], options: LogmOptions) -> List[np.ndarray]:
    """
    Compute matrix logarithm for a list of SPD covariance matrices.

    Args:
        covs: list of SPD matrices, each of shape (KC, KC).
        options: LogmOptions controlling the backend and parallelism.

    Returns:
        List of log(C) for each C in covs.
    """
    if options.mode == "scipy":
        fn = logm_spd_scipy
    elif options.mode == "eig":
        fn = lambda C: logm_spd_eig(C, eps=options.eig_eps)
    elif options.mode == "torch":
        fn = lambda C: logm_spd_torch(C, eps=options.eig_eps, device=options.torch_device)
    else:
        raise ValueError(f"Unknown logm mode: {options.mode}")

    if options.n_jobs and options.n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            outs = Parallel(n_jobs=options.n_jobs, backend="loky")(delayed(fn)(C) for C in covs)
            return outs
        except Exception:
            # Fallback to serial computation if joblib is unavailable or fails
            pass

    return [fn(C) for C in covs]


# -------------------------------------------------------------------------
# Temporal stacking and covariance construction
# -------------------------------------------------------------------------

def ntc_to_list_ct(X: np.ndarray) -> List[np.ndarray]:
    """
    Convert an array of shape (N, T, C) to a list of C×T matrices.

    This matches the behavior of the MATLAB helper that converts
    each trial from (T×C) to (C×T).

    Args:
        X: array of shape (N, T, C)

    Returns:
        A list of length N, where each element is a 2D array of shape (C, T).
    """
    X = np.asarray(X)
    assert X.ndim == 3, f"Expected X to have shape (N, T, C), got {X.shape}"
    N, T, C = X.shape
    out: List[np.ndarray] = []
    for i in range(N):
        Xi = X[i, :, :]  # (T, C)
        if Xi.shape != (T, C):
            Xi = Xi.reshape(T, C)
        out.append(Xi.T.copy())  # (C, T)
    return out


def _stack_delays(X_ct: np.ndarray, K: int, tau: int) -> np.ndarray:
    """
    Construct delay-embedded representation X_hat for a single trial.

    Args:
        X_ct: array of shape (C, T)
        K: number of delays
        tau: delay step in samples

    Returns:
        X_hat: array of shape (KC, T), where each block of C rows corresponds
               to a delayed version of X_ct, zero-padded on the left.
    """
    C, T = X_ct.shape
    KC = K * C
    X_hat = np.zeros((KC, T), dtype=np.float64)
    for k in range(K):
        n_delay = k * tau
        row_slice = slice(k * C, (k + 1) * C)
        if n_delay == 0:
            X_hat[row_slice, :] = X_ct
        else:
            # first n_delay columns are zero, remaining columns are shifted
            X_hat[row_slice, :n_delay] = 0.0
            X_hat[row_slice, n_delay:] = X_ct[:, : T - n_delay]
    return X_hat


def compute_covariance_train(
    X_list: List[np.ndarray],
    K: int,
    tau: int,
    logm_opt: LogmOptions,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute trace-normalized covariance and logm features for training trials.

    Args:
        X_list: list of length M; each entry is a matrix of shape (C, T)
                containing one trial.
        K: number of delays.
        tau: delay step in samples.
        logm_opt: options for SPD logm backend and parallelism.

    Returns:
        R:         (M, KC*KC) array; each row is vec(logm(C_k))^T.
        Cov_mean:  (KC, KC)   mean covariance (before logm) over training trials.
        cov_train: (KC, KC, M) stack of logm covariance matrices for all trials.
    """
    M = len(X_list)
    C, T = X_list[0].shape
    KC = K * C

    covs: List[np.ndarray] = []
    Sig_Cov = np.zeros((KC, KC), dtype=np.float64)
    for m in range(M):
        Xhat = _stack_delays(X_list[m], K, tau)   # (KC, T)
        Ck = Xhat @ Xhat.T                        # (KC, KC)
        tr = float(np.trace(Ck))
        if tr == 0.0:
            raise ValueError("trace(Ck) == 0; cannot normalize covariance.")
        Ck /= tr
        covs.append(Ck)
        Sig_Cov += Ck
    Cov_mean = Sig_Cov / M

    Ls = compute_logm_batch(covs, logm_opt)  # each is (KC, KC)
    R = np.zeros((M, KC * KC), dtype=np.float64)
    cov_train = np.zeros((KC, KC, M), dtype=np.float64)
    for m, Lm in enumerate(Ls):
        cov_train[:, :, m] = Lm
        R[m, :] = vecF(Lm)
    return R, Cov_mean, cov_train


def compute_covariance_test_nowhiten(
    X_list: List[np.ndarray],
    K: int,
    tau: int,
    logm_opt: LogmOptions,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute trace-normalized covariance and logm features for test trials.

    Args:
        X_list: list of length M; each entry is a matrix of shape (C, T)
        K: number of delays.
        tau: delay step in samples.
        logm_opt: options for SPD logm backend and parallelism.

    Returns:
        R_test: (M, KC*KC) array; each row is vec(logm(C_k))^T.
        cov_test: (KC, KC, M) stack of logm covariance matrices for all trials.
    """
    M = len(X_list)
    C, T = X_list[0].shape
    KC = K * C

    covs: List[np.ndarray] = []
    for m in range(M):
        Xhat = _stack_delays(X_list[m], K, tau)  # (KC, T)
        Ck = Xhat @ Xhat.T                       # (KC, KC)
        tr = float(np.trace(Ck))
        if tr == 0.0:
            raise ValueError("trace(Ck) == 0; cannot normalize covariance.")
        Ck /= tr
        covs.append(Ck)

    Ls = compute_logm_batch(covs, logm_opt)
    R_test = np.zeros((M, KC * KC), dtype=np.float64)
    cov_test = np.zeros((KC, KC, M), dtype=np.float64)
    for m, Lm in enumerate(Ls):
        cov_test[:, :, m] = Lm
        R_test[m, :] = vecF(Lm)
    return R_test, cov_test


# -------------------------------------------------------------------------
# Core SBLEST + CORAL algorithm
# -------------------------------------------------------------------------

@dataclass
class SBLESTParams:
    """
    Hyperparameters for SBLEST with CORAL alignment.

    Attributes:
        Maxiters: maximum number of SBLEST iterations.
        K: number of temporal delays.
        tau: delay step in samples.
        epsi: relative tolerance for convergence based on the loss change.
        logm_opt: LogmOptions controlling SPD logm computation.
    """
    Maxiters: int = 500
    K: int = 2
    tau: int = 1
    epsi: float = 1e-4
    logm_opt: LogmOptions = LogmOptions()


def SBLEST_CORAL_nowhiten(
    X_train: List[np.ndarray],
    Y: np.ndarray,
    X_test: List[np.ndarray],
    params: SBLESTParams,
):
    """
    SBLEST classifier with CORAL alignment in log-SPD space (no whitening of test).

    This is a port of the original MATLAB implementation, following the same
    steps:

        1) For each trial, construct delay-embedded covariance C_k from X_ct,
           trace-normalize, and apply matrix logarithm to obtain log(C_k).
        2) Compute mean log-covariance for source (train) and target (test),
           and derive a CORAL transform A_star in the log-SPD space.
        3) Apply A_star to the training log-covariances to obtain aligned
           cov_train_CORAL, then vectorize each into R (M×KC^2).
        4) Run the SBLEST optimization to estimate a symmetric classifier
           matrix W (KC×KC).
        5) Return:
             - W: classifier matrix
             - alpha: eigenvalues of W
             - V: eigenvectors of W
             - R_test: test features (vec(logm(C_test))) before alignment,
                       consistent with the original MATLAB interface.
             - cov_train_CORAL: aligned train log-covariances (KC×KC×M)
             - cov_test:       test log-covariances (KC×KC×M)

    Args:
        X_train: list of length N_train; each item is (C, T).
        Y: training labels of shape (N_train,). Must be in {-1, +1}.
        X_test: list of length N_test; each item is (C, T).
        params: SBLESTParams with K, tau, Maxiters, epsi, logm_opt.

    Returns:
        W: np.ndarray of shape (KC, KC)
        alpha: np.ndarray of eigenvalues of W
        V: np.ndarray of eigenvectors of W
        R_test: np.ndarray of shape (N_test, KC * KC)
        cov_train_CORAL: np.ndarray of shape (KC, KC, N_train)
        cov_test: np.ndarray of shape (KC, KC, N_test)
    """
    K, tau = params.K, params.tau
    Maxiters, e = params.Maxiters, params.epsi

    # 1) Log-covariance for train and test
    R_train, _, cov_train = compute_covariance_train(X_train, K, tau, params.logm_opt)
    R_test, cov_test = compute_covariance_test_nowhiten(X_test, K, tau, params.logm_opt)

    # 2) CORAL in log-SPD space
    C_S = np.mean(cov_train, axis=2)   # (KC, KC)
    C_T = np.mean(cov_test, axis=2)    # (KC, KC)

    rS = np.linalg.matrix_rank(C_S)
    rT = np.linalg.matrix_rank(C_T)
    r = min(rS, rT)

    # SVD (economy size), consistent with MATLAB svd(..., 'econ')
    US, sS, _ = np.linalg.svd(C_S, full_matrices=False)
    UT, sT, _ = np.linalg.svd(C_T, full_matrices=False)

    # Construct Sig = sqrtm(pinv(S_S)), but in diagonal form:
    inv_sqrt_sS = np.zeros_like(sS)
    nz = sS > 0
    inv_sqrt_sS[nz] = 1.0 / np.sqrt(sS[nz])
    Sig = np.diag(inv_sqrt_sS)

    # A_star = U_S * Sig * U_S' * U_T(:,1:r) * S_T(1:r,1:r)^(1/2) * U_T(:,1:r)'
    A_left = US @ Sig @ US.T
    UT_r = UT[:, :r]
    ST_r_sqrt = np.diag(np.sqrt(sT[:r]))
    A_star = A_left @ UT_r @ ST_r_sqrt @ UT_r.T

    KC = C_S.shape[0]
    N_train = cov_train.shape[2]

    cov_train_CORAL = np.zeros_like(cov_train)
    R = np.zeros((N_train, KC * KC), dtype=np.float64)
    AT = A_star.T
    for m in range(N_train):
        cov_train_CORAL[:, :, m] = (AT @ cov_train[:, :, m] @ A_star).real
        R[m, :] = vecF(cov_train_CORAL[:, :, m])

    # 3) SBLEST optimization
    M, D_R = R.shape
    if KC * KC != D_R:
        raise ValueError(f"Inconsistent dimensions: KC={KC}, feature dim={D_R}")

    # Symmetry check on each reshaped covariance
    for c in range(M):
        row_cov = R[c, :].reshape(KC, KC, order="F")
        if np.linalg.norm(row_cov - row_cov.T, ord="fro") > 1e-4:
            raise RuntimeError(f"Row {c+1} of R does not reshape to a symmetric matrix.")

    U = np.zeros((KC, KC), dtype=np.float64)  # classifier matrix W
    Psi = np.eye(KC, dtype=np.float64)
    lam = 1.0

    Loss_old = 0.0
    Yv = Y.reshape(-1, 1).astype(np.float64)
    I_M = np.eye(M, dtype=np.float64)

    import scipy.linalg as la

    for i in range(1, Maxiters + 1):
        # ---- MAP estimate of u (vectorized W) ----
        RPR = np.zeros((M, M), dtype=np.float64)
        B = np.zeros((KC * KC, M), dtype=np.float64)
        for c in range(KC):
            idx = slice(c * KC, (c + 1) * KC)
            Temp = Psi @ R[:, idx].T  # (KC, M)
            B[idx, :] = Temp
            RPR += R[:, idx] @ Temp   # (M, M)

        Sigma_y = RPR + lam * I_M     # (M, M)
        beta = la.solve(
            Sigma_y,
            Yv,
            assume_a="pos",
            check_finite=True,
        )                             # (M, 1)
        u = (B @ beta).reshape(-1)    # (KC^2,)
        U = u.reshape(KC, KC, order="F")
        U = 0.5 * (U + U.T)           # enforce symmetry

        # ---- Update Phi_c and Psi ----
        SR = la.solve(
            Sigma_y,
            R,
            assume_a="pos",
            check_finite=True,
        )                             # (M, KC^2)

        Phi_list: List[np.ndarray] = []
        for c in range(KC):
            idx = slice(c * KC, (c + 1) * KC)
            mid = R[:, idx].T @ SR[:, idx]      # (KC, KC)
            Phi_c = Psi - Psi @ mid @ Psi       # (KC, KC)
            Phi_list.append(Phi_c)

        PHI = np.zeros((KC, KC), dtype=np.float64)
        UU = np.zeros((KC, KC), dtype=np.float64)
        for c in range(KC):
            PHI += Phi_list[c]
            UU += np.outer(U[:, c], U[:, c])
        # symmetrize and average
        Psi = ((UU + UU.T) * 0.5 + (PHI + PHI.T) * 0.5) / KC

        # ---- Update lambda ----
        theta = 0.0
        for c in range(KC):
            idx = slice(c * KC, (c + 1) * KC)
            RtRt = R[:, idx].T @ R[:, idx]
            theta += np.trace(Phi_list[c] @ RtRt)

        resid = Yv - (R @ u.reshape(-1, 1))
        lam = (float((resid ** 2).sum()) + theta) / M

        # ---- Convergence check ----
        quad = float(Yv.T @ beta)            # Y' * inv(Sigma_y) * Y
        Loss = quad + calculate_log_det(Sigma_y)
        delta = abs(Loss - Loss_old) / max(1e-12, abs(Loss_old))
        if delta < e:
            break
        Loss_old = Loss

    # Eigen-decomposition of W (symmetric)
    evals, evecs = np.linalg.eigh(U)
    alpha = evals
    V = evecs
    W = U

    return W, alpha, V, R_test, cov_train_CORAL, cov_test

def generate_toy_data(
    n_train: int = 80,
    n_test: int = 40,
    n_channels: int = 10,
    n_timepoints: int = 200,
    seed: int = 0,
):
    """
    Generate small random data for a toy SBLEST demo.

    Shapes:
        X_train_ntc: (n_train, T, C)
        X_test_ntc:  (n_test,  T, C)
        y_train:     (n_train,) in {-1, +1}
        y_test:      (n_test,)  in {-1, +1}
    """
    rng = np.random.default_rng(seed)

    # Random time series: N × T × C
    X_train_ntc = rng.standard_normal((n_train, n_timepoints, n_channels))
    X_test_ntc = rng.standard_normal((n_test, n_timepoints, n_channels))

    # Random binary labels in {0, 1}, then standardized to {-1, +1}
    y_train_raw = rng.integers(0, 2, size=n_train)
    y_test_raw = rng.integers(0, 2, size=n_test)

    y_train = standardize_labels(y_train_raw)
    y_test = standardize_labels(y_test_raw)

    return X_train_ntc, y_train, X_test_ntc, y_test


def main():
    # 1) Generate toy data
    X_train_ntc, y_train, X_test_ntc, y_test = generate_toy_data()

    # Convert (N, T, C) -> List of (C, T)
    Xtr_list = ntc_to_list_ct(X_train_ntc)
    Xte_list = ntc_to_list_ct(X_test_ntc)

    # 2) Configure SBLEST parameters
    # Small dimensions here: C=10, K=2 -> KC=20, so logm on 20×20 is cheap.
    logm_opt = LogmOptions(
        mode="scipy",
        n_jobs=0,
    )
    params = SBLESTParams(
        Maxiters=2000,
        K=2,
        tau=1, # by Cross-validation
        epsi=1e-4,
        logm_opt=logm_opt,
    )

    # 3) Run SBLEST + CORAL
    W, alpha, V, R_test, cov_train_CORAL, cov_test = SBLEST_CORAL_nowhiten(
        Xtr_list, y_train, Xte_list, params
    )

    # 4) Simple linear decision using W (consistent with original MATLAB code)
    u = vecF(W)                  # (KC^2,)
    scores = R_test @ u          # (N_test,)
    y_pred = np.sign(scores)     # {-1, 0, +1}
    y_pred[y_pred == 0] = 1.0    # map 0 to +1 as in original implementation

    acc = float((y_pred.ravel() == y_test.ravel()).mean())

    # 5) Print summary
    print("\n=== SBLEST toy demo finished ===")
    print(f"Number of training trials: {len(Xtr_list)}")
    print(f"Number of test trials:     {len(Xte_list)}")
    print(f"W shape:                   {W.shape}")
    print(f"R_test shape:              {R_test.shape}")
    print(f"classification acc:    {acc:.4f}")


if __name__ == "__main__":
    main()