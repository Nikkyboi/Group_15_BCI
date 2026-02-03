import numpy as np

def cov_trace_normalized(X_trial):
    """
    X_trial: (C, T)
    Return: (C, C) covariance normalized by trace
    """
    C = X_trial @ X_trial.T
    return C / (np.trace(C) + 1e-12)

def mean_class_covariances(X, y):
    """
    X: (N, C, T)
    y: (N,)
    Return: Sigma0, Sigma1
    """
    X0 = X[y == 0]
    X1 = X[y == 1]

    Sigma0 = np.mean([cov_trace_normalized(tr) for tr in X0], axis=0)
    Sigma1 = np.mean([cov_trace_normalized(tr) for tr in X1], axis=0)
    return Sigma0, Sigma1

def whitening_matrix(Sigma_c):
    """
    Sigma_c: (C, C)
    Return: P (C, C) whitening matrix
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma_c)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    P = np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
    return P

def fit_csp_filters(X_train, y_train):
    """
    X_train: (N, C, T)
    y_train: (N,)
    Return: W_full (C, C)
    """
    Sigma0, Sigma1 = mean_class_covariances(X_train, y_train)
    Sigma_c = Sigma0 + Sigma1

    P = whitening_matrix(Sigma_c)

    S0 = P @ Sigma0 @ P.T
    eigvals, B = np.linalg.eigh(S0)

    idx = np.argsort(eigvals)[::-1]
    B = B[:, idx]

    W_full = B.T @ P  # (C, C)
    return W_full

def select_csp_filters(W_full, n_components=4):
    """
    W_full: (C, C)
    Return: W (n_components, C) using top+bottom eigenfilters
    """
    if n_components % 2 != 0:
        raise ValueError("n_components must be even: 2,4,6,...")

    m = n_components // 2
    W = np.concatenate([W_full[:m], W_full[-m:]], axis=0)
    return W

def csp_logvar_features(X, W):
    """
    X: (N, C, T)
    W: (K, C)
    Return: (N, K) log-variance features
    """
    Z = np.matmul(W[None, :, :], X)  # (N, K, T)
    var = np.var(Z, axis=2)          # (N, K)
    return np.log(var + 1e-12)

def fit_csp(X_train, y_train, n_components=4):
    W_full = fit_csp_filters(X_train, y_train)
    W = select_csp_filters(W_full, n_components=n_components)
    return W

if __name__ == "__main__":
    """
    CSP = Common Spatial Patterns
    
    What it does:
    - Spatial filtering method for EEG data
    - Finds spatial filters that maximize variance for one class while minimizing for the other
    - Enhances discriminative information in EEG signals for classification tasks (e.g., motor imagery)
    
    Steps:
    1. Compute class-specific covariance matrices
    2. Compute composite covariance matrix
    3. Whitening transformation
    4. Solve eigenvalue problem
    5. Select spatial filters
    6. Feature extraction (log-variance of filtered signals)
    
    """
    pass