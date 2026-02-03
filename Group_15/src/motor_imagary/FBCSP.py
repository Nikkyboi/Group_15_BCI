import numpy as np
from scipy.signal import butter, sosfiltfilt
from .CSP import fit_csp, csp_logvar_features

class FBCSP:
    """
    Filter Bank CSP feature extractor.

    Steps:
      1) For each frequency band:
          - bandpass filter EEG
          - fit CSP filters (W_band)
          - extract CSP log-variance features
      2) concatenate all band features into one vector

    After fit():
      transform(X) returns features shape (N, n_bands * n_components)
    """
    
    def __init__(self, fs, bands=None, n_components=4, order=4):
        """
        fs: sampling frequency
        bands: list of tuples [(8,12), (12,16), ...]
        n_components: CSP components per band (must be even for your CSP)
        """
        self.fs = fs
        self.bands = bands if bands is not None else [
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 24),
            (24, 28),
            (28, 30),
        ]
        self.n_components = n_components
        self.order = order

        self.W_per_band = []  # list of CSP projection matrices per band
        
    def fit(self, X_train, y_train):
        """
        X_train: (N, C, T)
        y_train: (N,)
        """
        self.W_per_band = []

        for band in self.bands:
            X_band = bandpass_filter(X_train, fs=self.fs, band=band, order=self.order)
            W = fit_csp(X_band, y_train, n_components=self.n_components)
            self.W_per_band.append(W)

        return self

    def transform(self, X):
        """
        X: (N, C, T)
        Return: (N, n_bands * n_components)
        """
        if len(self.W_per_band) == 0:
            raise RuntimeError("FBCSP not fitted. Call fit() before transform().")

        feats = []
        for band, W in zip(self.bands, self.W_per_band):
            X_band = bandpass_filter(X, fs=self.fs, band=band, order=self.order)
            F_band = csp_logvar_features(X_band, W)  # (N, n_components)
            feats.append(F_band)

        return np.concatenate(feats, axis=1)

    def fit_transform(self, X_train, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train)

def bandpass_filter(X, fs, band=(8, 12), order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    """
    low, high = band
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=2) # filter along time axis (axis=2 because X is (N,C,T))

if __name__ == "__main__":
    pass