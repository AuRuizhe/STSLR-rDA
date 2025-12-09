# file: main_stslr_rda.py
import numpy as np
from domain_align import domain_align
from log_L21_L1 import Class_L21L1


def main():
    # ======== 1. Synthetic example data (N, T, C) ==========
    # N_train / N_test: number of trials
    # T: number of time samples per trial
    # C: number of EEG channels
    n_trials_train, n_trials_test = 50, 20
    n_samples, n_channels = 1000, 62   # T, C

    # Raw EEG-like time series: shape = (N, T, C)
    dataTrain = np.random.randn(n_trials_train, n_samples, n_channels)
    dataTest = np.random.randn(n_trials_test, n_samples, n_channels)

    # Binary labels in {0, 1}
    y_train = np.random.choice([0, 1], size=n_trials_train)
    y_test = np.random.choice([0, 1], size=n_trials_test)

    # ======== 2. Domain alignment: time-delay + log-SPD + CORAL ==========
    # RsTrain, RsTest: (N_train, M, M) and (N_test, M, M)
    # where M = K * C after time-delay concatenation (K=2 when tau_delay=1).
    RsTrain, RsTest = domain_align(
        dataTrain,
        dataTest,
        tau_delay=1,
        eps=1e-9,
        shrink_Bg_to_I=0.2,
    )

    # ======== 3. Train STSLR classifier (logistic + L21+L1) ==========
    lam_l1 = 1e-3
    lam_l21 = 3e-3 # chosen by Cross-validation
    clf = Class_L21L1(
        lam_l1=lam_l1,
        lam_l21=lam_l21,
        max_iter=10000,
        tol=1e-5,
        verbose=0,
    )
    clf.fit(RsTrain, y_train)

    # ======== 4. Predict and evaluate ==========
    acc = clf.score(RsTest, y_test)
    print(f"Test accuracy (synthetic data): {acc:.4f}")


if __name__ == '__main__':
    main()
