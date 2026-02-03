# CSP_visualization.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt

# Your project imports
from .data import MI_EEG_Dataset  # from data.py
from .CSP import fit_csp, csp_logvar_features  # from CSP.py
from .rLDA import rLDA 

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_trials(subject: str, raw_path="data/Raw_data", processed_path="data/Processed"):
    """
    Loads trials as numpy:
      X: (N, T, C)
      y: (N,)
      fs: int
    """
    ds = MI_EEG_Dataset(
        subject=subject,
        raw_path=Path(raw_path),
        processed_path=Path(processed_path),
    )
    X = np.asarray(ds.X, dtype=np.float32)  # (N, T, C)
    y = np.asarray(ds.y, dtype=np.int64)
    fs = int(ds.fs) if ds.fs is not None else 256  # fallback
    return X, y, fs


def bandpass(X: np.ndarray, fs: int, band=(8, 30), order=4):
    """
    X: (N, T, C)
    """
    nyq = 0.5 * fs
    low = band[0] / nyq
    high = band[1] / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=1)


def quick_preprocess(
    X: np.ndarray,
    fs: int,
    crop_window_s=(0.5, 5.0),
    apply_notch=True,
    apply_bandpass=True,
    apply_car=True,
):
    """
    Minimal version matching your pipeline conceptually:
      - crop inside MI window
      - DC removal
      - 50 Hz notch
      - bandpass 8-30
      - CAR
    Input/Output: (N, T, C)
    """
    Xp = X.copy()

    # Crop
    start = int(crop_window_s[0] * fs)
    end = int(crop_window_s[1] * fs)
    end = min(end, Xp.shape[1])
    Xp = Xp[:, start:end, :]

    # Remove DC offset
    Xp = Xp - Xp.mean(axis=1, keepdims=True)

    # Notch 50 Hz
    if apply_notch:
        b, a = iirnotch(w0=50.0, Q=30.0, fs=fs)
        Xp = filtfilt(b, a, Xp, axis=1)

    # Bandpass 8-30 Hz
    if apply_bandpass:
        Xp = bandpass(Xp, fs, band=(8, 30), order=4)

    # CAR
    if apply_car:
        Xp = Xp - Xp.mean(axis=2, keepdims=True)

    return Xp


def to_csp_shape(X: np.ndarray) -> np.ndarray:
    """
    Convert from (N, T, C) -> (N, C, T) for your CSP.py functions
    """
    return np.transpose(X, (0, 2, 1))


def mean_covariance(X_ct: np.ndarray) -> np.ndarray:
    """
    X_ct: (N, C, T)
    Returns mean covariance (C, C) using simple cov estimate.
    """
    covs = []
    for i in range(X_ct.shape[0]):
        Xi = X_ct[i]  # (C, T)
        Xi = Xi - Xi.mean(axis=1, keepdims=True)
        Ci = (Xi @ Xi.T) / (Xi.shape[1] + 1e-12)
        covs.append(Ci)
    return np.mean(covs, axis=0)


# -----------------------------
# Plotting
# -----------------------------
def plot_timeseries_before_after_csp(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    fs: int,
    out_dir: Path,
    trial_idx: int = 0,
    n_raw_channels: int = 4,
):
    """
    Shows a single trial:
      - raw EEG channels
      - CSP components
    """
    ensure_dir(out_dir)

    # pick a trial index safely
    trial_idx = int(np.clip(trial_idx, 0, X_ct.shape[0] - 1))
    Xi = X_ct[trial_idx]  # (C, T)
    label = int(y[trial_idx])

    # Raw: first few channels
    C, T = Xi.shape
    ch_idx = np.arange(min(n_raw_channels, C))
    t = np.arange(T) / fs

    # CSP-projected components: Z = W @ X
    # W: (K, C), X: (C, T) -> (K, T)
    Z = (W @ Xi)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].set_title(f"Before CSP: Raw EEG channels (trial {trial_idx}, class={label})")
    for k in ch_idx:
        axes[0].plot(t, Xi[k], label=f"Ch {k}")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].set_title("After CSP: CSP components")
    for k in range(Z.shape[0]):
        axes[1].plot(t, Z[k], label=f"CSP comp {k}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Component amplitude")
    axes[1].legend(loc="upper right", ncol=2)

    fig.tight_layout()
    save_path = out_dir / "timeseries_before_after_csp.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_csp_feature_scatter(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
):
    """
    Plots log-variance features from CSP (N, K) using first two comps.
    """
    ensure_dir(out_dir)

    feats = csp_logvar_features(X_ct, W)  # (N, K)
    K = feats.shape[1]

    if K < 2:
        raise ValueError("Need at least 2 CSP components to make a 2D scatter plot.")

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    idx0 = (y == 0)
    idx1 = (y == 1)

    ax.scatter(feats[idx0, 0], feats[idx0, 1], label="Class 0 (Right)", alpha=0.8)
    ax.scatter(feats[idx1, 0], feats[idx1, 1], label="Class 1 (Left)", alpha=0.8)

    ax.set_title("CSP features (log-variance) — separability")
    ax.set_xlabel("Feature 1 (CSP comp 0 log-var)")
    ax.set_ylabel("Feature 2 (CSP comp 1 log-var)")
    ax.legend()

    fig.tight_layout()
    save_path = out_dir / "csp_feature_scatter.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_covariances_before_after(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
):
    """
    Plots mean class covariance matrices before CSP (C x C),
    and after CSP (K x K).
    """
    ensure_dir(out_dir)

    X0 = X_ct[y == 0]
    X1 = X_ct[y == 1]

    cov0 = mean_covariance(X0)
    cov1 = mean_covariance(X1)

    # After CSP: project to Z = W X -> covariance in CSP space
    Z0 = np.matmul(W[None, :, :], X0)  # (N0, K, T)
    Z1 = np.matmul(W[None, :, :], X1)  # (N1, K, T)

    cov0_csp = mean_covariance(Z0)
    cov1_csp = mean_covariance(Z1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    im00 = axes[0, 0].imshow(cov0, aspect="auto")
    axes[0, 0].set_title("Before CSP: Mean cov (Class 0)")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(cov1, aspect="auto")
    axes[0, 1].set_title("Before CSP: Mean cov (Class 1)")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im10 = axes[1, 0].imshow(cov0_csp, aspect="auto")
    axes[1, 0].set_title("After CSP: Mean cov (Class 0) in CSP space")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im11 = axes[1, 1].imshow(cov1_csp, aspect="auto")
    axes[1, 1].set_title("After CSP: Mean cov (Class 1) in CSP space")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("Index")
        ax.set_ylabel("Index")

    fig.tight_layout()
    save_path = out_dir / "covariances_before_after_csp.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


# -----------------------------
# Main runner
# -----------------------------
def run_visualization(
    subject: str = "PAT021_A_processed",
    n_components: int = 4,
    trial_idx: int = 0,
    raw_path: str = "data/Raw_data",
    processed_path: str = "data/Processed",
    do_preprocess: bool = True,
    out_dir: str = "Figures/CSP_vis",
):
    """
    Generates plots comparing before vs after CSP.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # Load
    X_ntc, y, fs = load_trials(subject = subject, raw_path=raw_path, processed_path=processed_path)
    y = remap_labels_to_01(y)
    # Preprocess for cleaner CSP (recommended)
    if do_preprocess:
        X_ntc = quick_preprocess(
            X_ntc,
            fs=fs,
            crop_window_s=(0.5, 5.0),
            apply_notch=True,
            apply_bandpass=True,
            apply_car=True,
        )

    # Convert to CSP shape
    X_nct = to_csp_shape(X_ntc)  # (N, C, T)

    # Fit CSP on whole subject (simple visualization purpose)
    W = fit_csp(X_nct, y, n_components=n_components)  # (K, C)
    
    print(f"[OK] Loaded subject={subject} | X={X_nct.shape} | fs={fs}")
    print(f"[OK] CSP W shape = {W.shape} (K, C)")

    p1 = plot_timeseries_before_after_csp(
        X_nct, y, W, fs, out_dir=out_dir, trial_idx=trial_idx, n_raw_channels=4
    )
    p2 = plot_csp_feature_scatter(X_nct, y, W, out_dir=out_dir)
    p3 = plot_covariances_before_after(X_nct, y, W, out_dir=out_dir)

    p4 = plot_1d_csp_feature_jitter(
    X_nct, y, W,
    out_dir=out_dir,
    feature_idx=0,   # try 0 or 1
    title="CSP feature separation (1D)",
    save_name="csp_feature_1d_jitter.png",
    )
    
    p = plot_1d_csp_feature_jitter_clean(
        X_nct, y, W,
        out_dir=out_dir,
        subject=subject,
        feature_idx=None
    )
    
    p = plot_1d_csp_feature_with_rlda_manual(
        X_nct, y, W,
        out_dir=out_dir,
        subject=subject,
        feature_idx=None,
        lam=0.1,
    )
    print("Saved:", p)
    
    p2d = plot_2d_csp_features(
        X_nct, y, W,
        out_dir=out_dir,
        subject=subject,
        feat_x=4,
        feat_y=0,  # or 1 or last feature
        save_name="csp_feature_2d_4_vs_0.png",
    )
    print("Saved:", p2d)
    

    print("\nSaved plots:")
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")
    print(f" - {p4}")
    print(f" - {p}")
    
def plot_1d_csp_feature_jitter(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
    feature_idx: int = 0,
    title: str = "1D CSP feature scatter (with jitter)",
    save_name: str = "csp_feature_1d_jitter.png",
):
    """
    Recreates the classic "clean CSP feature plot":
      x = one CSP log-variance feature
      y = random jitter just to spread points visually
    """
    ensure_dir(out_dir)

    feats = csp_logvar_features(X_ct, W)  # (N, K)
    x = feats[:, feature_idx]

    # random jitter on y-axis (purely visual)
    rng = np.random.default_rng(42)
    jitter0 = 0.2 + 0.35 * rng.random(np.sum(y == 0))
    jitter1 = 0.2 + 0.35 * rng.random(np.sum(y == 1))

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.scatter(x[y == 0], jitter0, marker="x", label="class 0", alpha=0.9)
    ax.scatter(x[y == 1], jitter1, marker="o", facecolors="none", label="class 1", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel(f"CSP feature (log-var), feature #{feature_idx}")
    ax.set_ylabel("jitter (visual only)")
    ax.set_yticks([])  # hide jitter axis ticks like the example
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_path = out_dir / save_name
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_1d_csp_feature_jitter_clean(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
    subject: str = "",
    feature_idx: int | None = None,
    save_name: str = "csp_feature_1d_jitter.png",
):
    ensure_dir(out_dir)

    feats = csp_logvar_features(X_ct, W)  # (N, K)

    # auto-pick best feature if not provided
    if feature_idx is None:
        feature_idx = pick_best_csp_feature(feats, y)

    x = feats[:, feature_idx]

    rng = np.random.default_rng(42)
    # two horizontal strips (like the paper-style plot)
    y0 = 2.0 + 0.25 * rng.standard_normal(np.sum(y == 0))
    y1 = 3.0 + 0.25 * rng.standard_normal(np.sum(y == 1))

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.scatter(x[y == 0], y0, marker="x", label="class 0", alpha=0.85)
    ax.scatter(
        x[y == 1], y1,
        marker="o",
        edgecolors="black",
        facecolors="none",
        linewidths=1.5,
        label="class 1",
        alpha=0.85,
    )

    ax.set_title(f"{subject} — CSP feature separation (1D) — best feature = #{feature_idx}")
    ax.set_xlabel("CSP feature (log-var)")
    ax.set_ylabel("")  # y-axis is visual only
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_path = out_dir / save_name
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def pick_best_csp_feature(feats: np.ndarray, y: np.ndarray) -> int:
    """
    Picks the CSP feature with best class separation using Fisher score:
    (mean0 - mean1)^2 / (var0 + var1)
    """
    best_idx = 0
    best_score = -1.0

    for k in range(feats.shape[1]):
        x0 = feats[y == 0, k]
        x1 = feats[y == 1, k]

        if len(x0) < 2 or len(x1) < 2:
            continue

        m0, m1 = np.mean(x0), np.mean(x1)
        v0, v1 = np.var(x0) + 1e-12, np.var(x1) + 1e-12
        score = (m0 - m1) ** 2 / (v0 + v1)

        if score > best_score:
            best_score = score
            best_idx = k

    return best_idx

def remap_labels_to_01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).squeeze()

    # Convert to int if possible
    if y.dtype.kind not in ("i", "u"):
        try:
            y = y.astype(int)
        except Exception:
            # fallback: map any unique values to 0/1
            uniq = np.unique(y)
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return np.vectorize(mapping.get)(y)

    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 classes, got {uniq} (counts={np.unique(y, return_counts=True)})")

    # Map smallest -> 0, largest -> 1
    mapping = {uniq[0]: 0, uniq[1]: 1}
    return np.vectorize(mapping.get)(y).astype(int)

def plot_1d_csp_feature_with_rlda_manual(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
    subject: str = "",
    feature_idx: int | None = None,
    lam: float = 0.1,
    save_name: str = "csp_feature_1d_with_rlda.png",
):
    """
    1D CSP feature + jitter + manual rLDA threshold.
    Works even if your rLDA.py crashes for 1D.
    """
    ensure_dir(out_dir)

    feats = csp_logvar_features(X_ct, W)  # (N, K)

    if feature_idx is None:
        feature_idx = pick_best_csp_feature(feats, y)

    x = feats[:, feature_idx]  # (N,)

    # --- manual rLDA in 1D ---
    x0 = x[y == 0]
    x1 = x[y == 1]
    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    v0, v1 = float(np.var(x0, ddof=1)), float(np.var(x1, ddof=1))

    # pooled variance
    n0, n1 = len(x0), len(x1)
    sp2 = ((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1)

    # shrinkage toward identity variance (=1 in 1D)
    sp2_reg = (1 - lam) * sp2 + lam * 1.0

    # rLDA weight and bias (equal priors)
    w = (m1 - m0) / (sp2_reg + 1e-12)
    b = -0.5 * (m1 + m0) * w

    # decision threshold x = -b/w
    x_thr = -b / (w + 1e-12)

    # --- jitter plot ---
    rng = np.random.default_rng(42)
    y0 = 2.0 + 0.25 * rng.standard_normal(np.sum(y == 0))
    y1 = 3.0 + 0.25 * rng.standard_normal(np.sum(y == 1))

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.scatter(x[y == 0], y0, marker="x", label="class 0", alpha=0.85)
    ax.scatter(
        x[y == 1], y1,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="class 1",
        alpha=0.85,
    )

    ax.axvline(x_thr, linestyle="--", linewidth=2, label=f"rLDA boundary (λ={lam})")

    ax.set_title(f"{subject} — CSP feature (1D) + rLDA boundary — feature #{feature_idx}")
    ax.set_xlabel("CSP feature (log-var)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_path = out_dir / save_name
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_2d_csp_features(
    X_ct: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    out_dir: Path,
    subject: str = "",
    feat_x: int = 0,
    feat_y: int = 1,
    save_name: str = "csp_feature_2d.png",
):
    """
    Real 2D plot: CSP feature feat_x vs CSP feature feat_y
    """
    ensure_dir(out_dir)

    feats = csp_logvar_features(X_ct, W)  # (N, K)

    x = feats[:, feat_x]
    yv = feats[:, feat_y]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.scatter(x[y == 0], yv[y == 0], marker="x", label="class 0", alpha=0.85)
    ax.scatter(
        x[y == 1], yv[y == 1],
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="class 1",
        alpha=0.85,
    )

    ax.set_title(f"{subject} — CSP features in 2D (#{feat_x} vs #{feat_y})")
    ax.set_xlabel(f"CSP feature #{feat_x} (log-var)")
    ax.set_ylabel(f"CSP feature #{feat_y} (log-var)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_path = out_dir / save_name
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    # Simple way to run without extra dependencies:
    # python CSP_visualization.py
    #
    # Change subject here if needed:
    run_visualization(
        subject="PAT021_A_processed",
        n_components=6,
        trial_idx=0,
        do_preprocess=True,
        out_dir="reports/CSP_vis",
    )
    
