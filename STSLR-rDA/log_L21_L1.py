# file: log_l21_l1.py
import numpy as np
from scipy.special import expit     # numerically stable sigmoid


def logistic_loss(W, b, X2, y):
    """Logistic loss with binary labels in {0, 1}.

    Parameters
    ----------
    W : array, shape (m, m)
        Weight matrix in the latent SPD space.
    b : float
        Bias term.
    X2 : array, shape (n, m*m)
        Each row is vec(R_i), where R_i is an (m, m) SPD/log-SPD matrix.
    y : array, shape (n,)
        Binary labels in {0, 1}.

    Returns
    -------
    loss : float
        Sum of logistic cross-entropy over all samples.
    """
    z = X2 @ W.reshape(-1) + b            # (n,)
    hx = expit(z)
    eps = 1e-12
    return -np.sum(y * np.log(hx + eps) + (1 - y) * np.log(1 - hx + eps))


def grad_W_b(W, b, X2, y):
    """Gradient of logistic loss w.r.t. W and b (averaged over samples)."""
    n = y.size
    z = X2 @ W.reshape(-1) + b            # (n,)
    h = expit(z)                          # (n,)
    diff = (h - y) / n                    # (n,)

    g_b = diff.sum()                      # scalar
    g_W = (diff[:, None] * X2).sum(0)     # (m*m,)
    return g_W.reshape(W.shape), g_b


def prox_L1_L21(V, step, lam_l1, lam_l21):
    """Composite proximal operator for L1 + L21 regularization.

    First apply element-wise L1 soft-thresholding, then apply
    L21 shrinkage across rows.

    V : (m, m)
    """
    # L1 soft-thresholding
    V = np.sign(V) * np.maximum(np.abs(V) - step * lam_l1, 0.0)

    # Group L21 on rows
    row_norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-16
    coeff = np.maximum(0.0, 1 - step * lam_l21 / row_norm)
    return V * coeff


def train_LogL21_L1(X, y, lam_l1=1e-5, lam_l21=1e-4,
                    max_iter=10000, tol=1e-5, verbose=50):
    """APGM solver for sparse logistic regression on SPD features.

    We minimize:
        sum_i log(1 + exp(-y_i * w^T x_i)) + lambda_21 * ||W||_{21}
                                         + lambda_1  * ||W||_1
    where x_i = vec(R_i), and R_i is an SPD/log-SPD matrix.

    Parameters
    ----------
    X : array, shape (n, m, m)
        SPD/log-SPD matrices for n trials.
    y : array, shape (n,)
        Binary labels in {0, 1} or {-1, 1}.
    lam_l1 : float
        L1 regularization weight.
    lam_l21 : float
        L21 (row group) regularization weight.
    max_iter : int
        Maximum number of APGM iterations.
    tol : float
        Relative change tolerance for early stopping.
    verbose : int
        Print progress every `verbose` iterations (0 disables printing).

    Returns
    -------
    W : array, shape (m, m)
        Estimated weight matrix.
    b : float
        Estimated bias term.
    F_new : float
        Final objective value.
    k : int
        Number of iterations performed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    # ensure labels are 0/1
    unique = np.unique(y)
    if set(unique) == {-1.0, 1.0}:
        y = (y > 0).astype(float)
    elif not set(unique).issubset({0.0, 1.0}):
        raise ValueError("y must contain labels in {0, 1} or {-1, 1}")

    n, m, _ = X.shape
    X2 = X.reshape(n, m * m)              # flatten each SPD matrix

    # Lipschitz constant and fixed step size
    L = 0.25 * np.max(np.sum(X2 ** 2, axis=1))   # logistic: L = 1/4 * max ||x||^2
    step = 1.0 / L if L > 0 else 1.0

    # Initialization
    W = np.zeros((m, m))
    b = 0.0
    W_prev, b_prev = W.copy(), b
    F_prev = logistic_loss(W, b, X2, y) \
           + lam_l21 * np.linalg.norm(W, axis=1).sum() \
           + lam_l1 * np.abs(W).sum()
    alpha_prev = 1.0

    for k in range(1, max_iter + 1):
        # Nesterov extrapolation
        alpha = (1 + np.sqrt(1 + 4 * alpha_prev ** 2)) / 2
        beta = (alpha_prev - 1) / alpha
        P  = W + beta * (W - W_prev)
        Pb = b + beta * (b - b_prev)

        # Gradient at extrapolated point
        gW, gb = grad_W_b(P, Pb, X2, y)

        # Proximal step
        V = P - step * gW
        W_new = prox_L1_L21(V, step, lam_l1, lam_l21)
        b_new = Pb - step * gb

        # Objective
        F_new = logistic_loss(W_new, b_new, X2, y) \
              + lam_l21 * np.linalg.norm(W_new, axis=1).sum() \
              + lam_l1 * np.abs(W_new).sum()

        rel_drop = abs(F_prev - F_new) / max(1.0, F_prev)
        if verbose and k % verbose == 0:
            print(f'iter={k:4d},  F={F_new:.6g},  relΔ={rel_drop:.3e}')

        if rel_drop < tol:
            break

        # Prepare for next iteration
        W_prev, b_prev = W, b
        W, b           = W_new, b_new
        F_prev         = F_new
        alpha_prev     = alpha

    return W, b, F_new, k


class Class_L21L1:
    """Convenience wrapper around train_LogL21_L1.

    Usage:
        clf = Class_L21L1(lam_l1=..., lam_l21=...)
        clf.fit(Rs_train, y_train)
        acc = clf.score(Rs_test, y_test)
    """
    def __init__(self, lam_l1=1e-5, lam_l21=1e-4,
                 max_iter=10000, tol=1e-5, verbose=50):
        self.lam_l1 = lam_l1
        self.lam_l21 = lam_l21
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """Fit the sparse logistic regression model.

        X : (n, m, m)  SPD/log-SPD feature matrices
        y : (n,)       binary labels in {0, 1} or {-1, 1}
        """
        W, b, _, _ = train_LogL21_L1(
            X, y,
            lam_l1=self.lam_l1,
            lam_l21=self.lam_l21,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )
        self.W_ = W
        self.b_ = b
        return self

    def decision_function(self, X):
        """Return signed scores z = w^T x + b for each SPD matrix."""
        if self.W_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        n, m, _ = X.shape
        X2 = X.reshape(n, m * m)
        z = X2 @ self.W_.reshape(-1) + self.b_
        return z

    def predict_proba(self, X):
        """Return predicted probabilities P(y=1|x)."""
        z = self.decision_function(X)
        p1 = expit(z)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        """Return hard label predictions in {0, 1}."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """Return classification accuracy on given data."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
