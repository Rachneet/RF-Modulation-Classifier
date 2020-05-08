import numpy as np


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