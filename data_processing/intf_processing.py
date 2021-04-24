import numpy as np
from visualization.visualize import iq_to_complex
import data_processing.read_h5 as reader
from sklearn import preprocessing


def compute_ica(X, step_size=1, tol=1e-8, max_iter=10000, n_sources=2):
    m, n = X.shape

    # Initialize random weights
    W = np.random.rand(n_sources, m)

    for c in range(n_sources):
        # Copy weights associated to the component and normalize
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        for i in range(max_iter):
            # Dot product of weight and input
            v = np.dot(w.T, X)

            # Pass w*s into contrast function g
            gv = np.tanh(v * step_size).T

            # Pass w*s into g prime
            gdv = (1 - np.square(np.tanh(v))) * step_size

            # Update weights
            wNew = (X * gv.T).mean(axis=1) - gdv.mean() * w.squeeze()

            # Decorrelate and normalize weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)
            # Update weights
            w = wNew
            # Check for convergence
            if lim < tol:
                break

        W[c, :] = w.T
        return W


def featurize(sample):
    N = sample.size
    r = iq_to_complex(sample)
    # print(r)
    r_star = np.conj(r)
    # print(r_star)
    # zero-mean
    r10 = r - np.mean(r)
    r11 = r_star - np.mean(r_star)
    # Moments
    r20 = r10 * r10
    r21 = r10 * r11
    r22 = r11 * r11
    r40 = r20 * r20

    M20 = (1 / N) * np.sum(r20)
    M21 = (1 / N) * np.sum(r21)
    M22 = (1 / N) * np.sum(r22)
    M40 = (1 / N) * np.sum(r40)
    M41 = (1 / N) * np.sum(r20 * r21)
    M42 = (1 / N) * np.sum(r20 * r22)
    M60 = (1 / N) * np.sum(r40 * r20)
    M61 = (1 / N) * np.sum(r40 * r21)
    M62 = (1 / N) * np.sum(r40 * r22)
    M63 = (1 / N) * np.sum(r20 * r10 * r22 * r11)
    # Statistical Features
    C20 = np.abs(M20)
    C21 = np.abs(M21)

    mu_42 = np.abs(M42 / np.abs(M21))

    C40 = np.abs((-3 * M20 ** 2 + M40) / np.abs(M21) ** 2)
    C41 = np.abs((-3 * M21 * M20 + M41) / np.abs(M21) ** 2)
    C42 = np.abs((-2 * M21 ** 2 - M22 * M20 + M42) / np.abs(M21) ** 2)

    C60 = np.abs((30 * M20 ** 3 - 15 * M20 * M40 + M60) / np.abs(M21) ** 3)
    C61 = np.abs(
        (30 * M20 ** 2 * M21 - 14 * M20 * M41 - M40 * M21 + M61) / np.abs(M21) ** 3)
    C62 = np.abs(
        (24 * M21 ** 2 * M20 + 6 * M22 * M20 ** 2 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40 + M62) / np.abs(M21) ** 3)
    C63 = np.abs(
        (12 * M21 ** 3 + 12 * M22 * M21 * M20 - 9 * M42 * M21 + M63) / np.abs(M21) ** 3)

    return np.array([C20,mu_42,C21,C40,C41,C42,C60,C61,C62,C63])


if __name__=='__main__':
    path = "/media/backup/Arsenal/rf_dataset_inets/dataset_intf_bpsk_snr20_sir25_ext.h5"
    iq, labels, snrs = reader.read_hdf5(path)
    res = featurize(iq[0])
    res = preprocessing.scale(res,with_mean=False)
    print(res)