# file: main_stslr_rda.py
import os

import numpy as np
import scipy

from domain_align import domain_align
from log_L21_L1 import Class_L21L1


def main():
    # ======== 1. Synthetic example data (N, T, C) ==========
    # N_train / N_test: number of trials
    # T: number of time samples per trial
    # C: number of EEG channels
    # n_trials_train, n_trials_test = 50, 20
    # n_samples, n_channels = 1000, 62   # T, C
    #
    # # Raw EEG-like time series: shape = (N, T, C)
    # dataTrain = np.random.randn(n_trials_train, n_samples, n_channels)
    # dataTest = np.random.randn(n_trials_test, n_samples, n_channels)
    #
    # # Binary labels in {0, 1}
    # y_train = np.random.choice([0, 1], size=n_trials_train)
    # y_test = np.random.choice([0, 1], size=n_trials_test)

    data_path = 'D:\SEED情绪\Cross_data/'
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.mat')])
    accTest = []

    for k in range(1):
        fname = files[k]
        print(f"\n=== Subject {k}: {fname} ===")
        mat = scipy.io.loadmat(os.path.join(data_path, fname))
        X = mat['X_train']  # (Ns, T, C)
        X_test = mat['X_test']  # (Nt, T, C)
        labelTrain = mat['y_train'].ravel()
        labelTest = mat['y_test'].ravel()
        labelTrain[labelTrain == -1] = 0
        labelTest[labelTest == -1] = 0
        dataTrain = X
        dataTest = X_test
        y_train = labelTrain
        y_test = labelTest
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
    lam_l21 = 3e-3
    clf = Class_L21L1(
        lam_l1=lam_l1,
        lam_l21=lam_l21,
        max_iter=10000,
        tol=1e-5,
        verbose=200,
    )
    clf.fit(RsTrain, y_train)

    # ======== 4. Predict and evaluate ==========
    acc = clf.score(RsTest, y_test)
    print(f"Test accuracy (synthetic data): {acc:.4f}")


if __name__ == '__main__':
    main()
