# file: domain_align.py
import numpy as np
import scipy.linalg

# =========================
# Helper functions (SPD / vectorization)
# =========================

def _sym(X: np.ndarray) -> np.ndarray:
    """Symmetrize a square matrix."""
    return 0.5 * (X + X.T)


def _spd_shrink(S: np.ndarray, eps: float) -> np.ndarray:
    """Make a matrix numerically SPD by adding eps*I and symmetrizing."""
    return _sym(S) + eps * np.eye(S.shape[0])


def _sqrtm_spd(S: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root for an SPD matrix."""
    w, V = np.linalg.eigh(_sym(S))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _invsqrtm_spd(S: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root for an SPD matrix."""
    w, V = np.linalg.eigh(_sym(S))
    w = np.clip(w, 1e-30, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def triu_index(D: int):
    """Return indices of the upper triangular part (including diagonal)."""
    return np.triu_indices(D)


def sym_to_vec_upper(L: np.ndarray, tri_idx=None) -> np.ndarray:
    """Symmetric matrix -> upper-triangular vector (no sqrt(2) scaling)."""
    D = L.shape[0]
    if tri_idx is None:
        tri_idx = triu_index(D)
    return L[tri_idx]


def vec_upper_to_sym(v: np.ndarray, D: int, tri_idx=None) -> np.ndarray:
    """Upper-triangular vector -> symmetric matrix by mirroring."""
    if tri_idx is None:
        tri_idx = triu_index(D)
    M = np.zeros((D, D), dtype=float)
    M[tri_idx] = v
    i, j = tri_idx
    M[j, i] = v
    return _sym(M)


# =========================
# 1) Time-delay concatenation
# =========================

class DelayConcat:
    """Concatenate a time-delayed copy of the signal along the channel axis.

    Input:
        X: (N, T, C)

    Output:
        X_concat: (N, T, C)        if tau == 0
                  (N, T, 2*C)      if tau > 0 (original + delayed copy)
    """
    def __init__(self, tau: int = 1):
        self.tau = int(tau)

    def fit(self, X, y=None):
        assert X.ndim == 3, "X must be (N, T, C)"
        return self

    def transform(self, X):
        if self.tau == 0:
            return X
        N, T, C = X.shape
        Xs = np.zeros_like(X)
        if self.tau < T:
            Xs[:, self.tau:, :] = X[:, : T - self.tau, :]
        # concatenate original and delayed copy along the channel axis
        return np.concatenate([X, Xs], axis=2)  # (N, T, 2C)


# =========================
# 2) Covariance with trace normalization (no log yet)
# =========================

class CovTraceNoLog:
    """Per-trial covariance with trace normalization.

    Input:
        X: (N, T, D)

    Output:
        cov: (D, D, N)
    """
    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    def _cov_trace_norm(self, X: np.ndarray) -> np.ndarray:
        # X: (T, D) -> (D, D)
        S = _sym(X.T @ X)
        tr = float(np.trace(S))
        if tr <= self.eps:
            S = S + self.eps * np.eye(S.shape[0])
            tr = float(np.trace(S))
        return S / tr

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (N, T, D) -> cov: (D, D, N)
        N, T, D = X.shape
        out = np.zeros((D, D, N), dtype=np.float64)
        for i in range(N):
            S = self._cov_trace_norm(X[i])
            out[:, :, i] = S + self.eps * np.eye(D)
        return out


# =========================
# 3) Matrix logarithm on SPD covariance
# =========================

class LogmTransformer:
    """Matrix logarithm on SPD covariance matrices.

    Input:
        cov: (D, D, N)

    Output:
        log-cov: (D, D, N)
    """
    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    def fit(self, X, y=None):
        return self

    def transform(self, cov: np.ndarray) -> np.ndarray:
        D, _, N = cov.shape
        out = np.zeros((D, D, N), dtype=np.float64)
        I = np.eye(D)
        for i in range(N):
            M = _sym(cov[:, :, i]) + self.eps * I
            L = scipy.linalg.logm(M)
            out[:, :, i] = _sym(np.real(L))
        return out


# =========================
# 4) Global CORAL in log-SPD space (source -> target)
# =========================

class GlobalCORAL_LogSPD:
    """Global CORAL in log-SPD vector space (source -> target).

    Only source samples are transformed. Target samples are left unchanged.

    Given source and target log-covariance matrices:
      - Vectorize each trial using upper-triangular entries;
      - Estimate source and target means / covariances;
      - Compute a whitening-recoloring transform B_g that matches
        source covariance to target covariance, with optional
        shrinkage toward the identity (regularization);
      - Compute a shift m_g so that transformed source mean matches
        the target mean.

    Transformed source features are obtained by:
        x' = B_g x + m_g.
    """
    def __init__(self, eps: float = 1e-6,
                 shrink_Bg_to_I: float = 0.2,
                 verbose: bool = False):
        self.eps = float(eps)
        self.shrink_Bg_to_I = float(shrink_Bg_to_I)
        self.verbose = bool(verbose)

        self.Bg_ = None   # (p, p)
        self.mg_ = None   # (p,)
        self.tri_idx_ = None
        self.D_ = None
        self.p_ = None

    def fit(self, logC_src: np.ndarray, logC_tgt: np.ndarray):
        """Estimate global CORAL transform from source to target.

        logC_src: (Ns, D, D)  source log-covariance matrices
        logC_tgt: (Nt, D, D)  target log-covariance matrices
        """
        Ns, D, _ = logC_src.shape
        Nt = logC_tgt.shape[0]
        self.D_ = D
        tri_idx = triu_index(D)
        self.tri_idx_ = tri_idx
        p = len(tri_idx[0])
        self.p_ = p

        # Vectorize symmetric matrices
        Xs = np.zeros((Ns, p), dtype=float)
        for i in range(Ns):
            Xs[i] = sym_to_vec_upper(logC_src[i], tri_idx)
        Xt = np.zeros((Nt, p), dtype=float)
        for j in range(Nt):
            Xt[j] = sym_to_vec_upper(logC_tgt[j], tri_idx)

        # Global source statistics
        mu_s = Xs.mean(axis=0)
        Xs_c = Xs - mu_s
        Sig_s = _spd_shrink((Xs_c.T @ Xs_c) / max(Ns - 1, 1), self.eps)

        # Global target statistics
        mu_t = Xt.mean(axis=0)
        Xt_c = Xt - mu_t
        Sig_t = _spd_shrink((Xt_c.T @ Xt_c) / max(Nt - 1, 1), self.eps)

        # Whitening-recoloring: source -> target
        Bg = _sqrtm_spd(Sig_t) @ _invsqrtm_spd(Sig_s)

        # Regularization: shrink Bg toward identity
        if self.shrink_Bg_to_I > 0:
            lam = self.shrink_Bg_to_I
            Bg = lam * np.eye(p) + (1 - lam) * Bg

        mg = mu_t - Bg @ mu_s

        self.Bg_ = Bg
        self.mg_ = mg
        return self

    def transform_source(self, logC_src: np.ndarray) -> np.ndarray:
        """Apply the CORAL transform to source log-SPD matrices.

        logC_src: (Ns, D, D)
        Returns:
            (Ns, D, D)  transformed source log-SPD matrices
        """
        assert self.Bg_ is not None and self.mg_ is not None
        Ns, D, _ = logC_src.shape
        tri_idx = self.tri_idx_
        p = self.p_

        Xs = np.zeros((Ns, p), dtype=float)
        for i in range(Ns):
            Xs[i] = sym_to_vec_upper(logC_src[i], tri_idx)

        Xo = (Xs @ self.Bg_.T) + self.mg_[None, :]
        out = np.zeros((Ns, D, D), dtype=float)
        for i in range(Ns):
            out[i] = vec_upper_to_sym(Xo[i], D, tri_idx)
        return out


# =========================
# 5) From time series to log-SPD matrices
# =========================

def logspd_features_from_timeseries(Z: np.ndarray,
                                    eps: float = 1e-9) -> np.ndarray:
    """Convert time-series data to log-SPD covariance matrices.

    Input
    -----
    Z : (N, T, D)
        Time-series data (trials, time samples, channels).

    Output
    ------
    X_mat : (N, D, D)
        Log-covariance matrices per trial.
    """
    cov_builder = CovTraceNoLog(eps=eps)
    log_tf = LogmTransformer(eps=eps)
    cov_spd = cov_builder.transform(Z)      # (D, D, N)
    cov_log = log_tf.transform(cov_spd)     # (D, D, N)
    X_mat = np.transpose(cov_log, (2, 0, 1))  # (N, D, D)
    return X_mat


# =========================
# 6) Top-level domain alignment function
# =========================

def domain_align(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tau_delay: int = 1,
    eps: float = 1e-9,
    shrink_Bg_to_I: float = 0.2,
):
    """End-to-end domain alignment for STSLR-rDA.

    Steps:
      1) Time-delay concatenation along channels (K=2 with tau=1 by default);
      2) Per-trial covariance with trace normalization;
      3) Matrix logarithm to obtain log-SPD matrices;
      4) Global CORAL (source -> target) in log-SPD vector space, with
         shrinkage toward identity (regularized CORAL).

    Parameters
    ----------
    X_train : array, shape (N_train, T, C)
        Source-domain time-series data: trials x time samples x channels.
    X_test : array, shape (N_test, T, C)
        Target-domain time-series data: trials x time samples x channels.
    tau_delay : int, default=1
        Number of samples for the time delay in DelayConcat. If > 0,
        the delayed copy is concatenated along the channel axis.
    eps : float, default=1e-9
        Small constant for numerical stability in covariance and logm.
    shrink_Bg_to_I : float, default=0.2
        Shrinkage factor for the CORAL transform toward the identity
        (this is the "regularized" part of STSLR-rDA).

    Returns
    -------
    Rs_train : array, shape (N_train, M, M)
        Log-SPD matrices for the aligned source domain (after CORAL).
    Rs_test : array, shape (N_test, M, M)
        Log-SPD matrices for the target domain (no transform applied).
        Here M = K * C after time-delay concatenation.
    """
    assert X_train.ndim == 3 and X_test.ndim == 3, \
        "Inputs must be (N, T, C)"

    # 1) Time-delay concatenation
    dc = DelayConcat(tau=tau_delay)
    Zs = dc.fit(X_train).transform(X_train)  # (N_train, T, K*C)
    Zt = dc.transform(X_test)                # (N_test,  T, K*C)

    # 2–3) From time series to log-SPD covariance matrices
    Xs_mat = logspd_features_from_timeseries(Zs, eps=eps)  # (N_train, M, M)
    Xt_mat = logspd_features_from_timeseries(Zt, eps=eps)  # (N_test,  M, M)

    # 4) Global CORAL (source -> target; only source is transformed)
    g_align = GlobalCORAL_LogSPD(
        eps=1e-6,
        shrink_Bg_to_I=shrink_Bg_to_I,
        verbose=False,
    )
    g_align.fit(Xs_mat, Xt_mat)
    Xs_mat_aligned = g_align.transform_source(Xs_mat)

    return Xs_mat_aligned, Xt_mat
