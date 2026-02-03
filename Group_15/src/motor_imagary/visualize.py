from pathlib import Path
import scipy.io as sio
from torch.utils.data import Dataset
import typer
import matplotlib.pyplot as plt
import numpy as np 
import torch
from .data import MI_EEG_Dataset
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# needed data
TRIALS = {
    "PAT013": 494,
    "PAT015": 1126,
    "PAT021_A": 1236,
    "PATID15": 1960,
    "PATID16": 434,
    "PATID26": 1464,
}

# Table 2: Cross-subject (LOSO)
ACC_CROSS = {
    "CSP+rLDA (No EA)":   {"PAT013": 0.5405, "PAT015": 0.5115, "PAT021_A": 0.7387, "PATID15": 0.7281, "PATID16": 0.5415, "PATID26": 0.5512},
    "CSP+rLDA (+EA)":     {"PAT013": 0.6296, "PAT015": 0.6314, "PAT021_A": 0.7411, "PATID15": 0.7699, "PATID16": 0.5599, "PATID26": 0.5977},
    "FBCSP+rLDA (No EA)": {"PAT013": 0.6053, "PAT015": 0.5790, "PAT021_A": 0.6812, "PATID15": 0.7857, "PATID16": 0.5392, "PATID26": 0.5485},
    "FBCSP+rLDA (+EA)":   {"PAT013": 0.6498, "PAT015": 0.6341, "PAT021_A": 0.6699, "PATID15": 0.7658, "PATID16": 0.5369, "PATID26": 0.5745},
}

# Table 3: Within-subject
ACC_WITHIN = {
    "CSP+rLDA (No EA)":   {"PAT013": 0.6667, "PAT015": 0.7434, "PAT021_A": 0.8669, "PATID15": 0.8546, "PATID16": 0.5517, "PATID26": 0.5802},
    "CSP+rLDA (+EA)":     {"PAT013": 0.6768, "PAT015": 0.7301, "PAT021_A": 0.8669, "PATID15": 0.8495, "PATID16": 0.5747, "PATID26": 0.5939},
    "FBCSP+rLDA (No EA)": {"PAT013": 0.7475, "PAT015": 0.8230, "PAT021_A": 0.8831, "PATID15": 0.8622, "PATID16": 0.7241, "PATID26": 0.7543},
    "FBCSP+rLDA (+EA)":   {"PAT013": 0.6970, "PAT015": 0.8053, "PAT021_A": 0.8911, "PATID15": 0.8699, "PATID16": 0.7011, "PATID26": 0.7099},
}

def _plot_accuracy_vs_trials(trials_dict, acc_dict, title, save_path=None):
    subjects = list(trials_dict.keys())
    x = np.array([trials_dict[s] for s in subjects], dtype=float)

    plt.figure(figsize=(11, 7))

    # Plot each method
    for method_name, method_accs in acc_dict.items():
        y = np.array([method_accs[s] for s in subjects], dtype=float) * 100.0  # -> %

        # Pearson correlation
        R = np.corrcoef(x, y)[0, 1]

        plt.scatter(x, y, s=180, alpha=0.9, label=f"{method_name} (R={R:.2f})")

        # Label points
        for s in subjects:
            plt.text(
                trials_dict[s],
                method_accs[s] * 100.0 + 0.8,
                s.replace("_A", "").replace("PATID", "ID"),
                fontsize=10,
                ha="center",
            )

    plt.title(title, fontsize=16)
    plt.xlabel("Number of trials (data quantity)", fontsize=12)
    plt.ylabel("Classification accuracy (%)", fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", framealpha=0.95)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure -> {save_path}")

    plt.show()
    
def plot_cross_subject_accuracy_vs_trials(save_path=None):
    _plot_accuracy_vs_trials(
        trials_dict=TRIALS,
        acc_dict=ACC_CROSS,
        title="Cross-subject (LOSO): Trials vs Accuracy",
        save_path=save_path,
    )
    
def plot_within_subject_accuracy_vs_trials(save_path=None):
    _plot_accuracy_vs_trials(
        trials_dict=TRIALS,
        acc_dict=ACC_WITHIN,
        title="Within-subject: Trials vs Accuracy",
        save_path=save_path,
    )


def EA_alignment_visualized(
    subj_a="PAT013_processed",
    subj_b="PAT015_processed",
    processed_no_ea=Path("data/Processed"),
    processed_w_ea=Path("data/Processed_w_EA"),
    save_path=Path("reports/EA_visualization_PAT013_PAT015.png"),
    clip_percentiles=(1, 99),
    subsample_per_subject=400,
    random_state=0,
    trace_normalize=True,
):
    """
    Visualize Euclidean Alignment effect by comparing TWO subjects together.
    Each point = one trial (covariance features -> PCA 2D)
    Color = subject/domain (NOT class label).

    BEFORE: trials from Processed (No EA)
    AFTER : trials from Processed_w_EA (+EA)
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    def pretty_name(s: str) -> str:
        return s.replace("_processed", "")

    def load_subject_trials(subject: str, base_path: Path) -> np.ndarray:
        ds = MI_EEG_Dataset(subject=subject, processed_path=base_path)
        return ds.X.astype(np.float64)  # (N, T, C)

    def cov_features(X_ntc: np.ndarray) -> np.ndarray:
        """
        X_ntc: (N, T, C)
        Return: (N, D) where D=C*(C+1)/2 from upper triangle of covariance.
        Uses fast covariance estimate via einsum (no Python loop).
        """
        N, T, C = X_ntc.shape

        # center per trial (remove mean over time)
        X = X_ntc - X_ntc.mean(axis=1, keepdims=True)

        # covariance per trial: (N, C, C)
        # cov = (X^T X) / (T-1)
        cov = np.einsum("ntc,ntd->ncd", X, X) / max(T - 1, 1)

        if trace_normalize:
            tr = np.trace(cov, axis1=1, axis2=2)  # (N,)
            cov = cov / (tr[:, None, None] + 1e-12)

        iu = np.triu_indices(C)
        return cov[:, iu[0], iu[1]]  # (N, D)

    def clip_outliers(F: np.ndarray, lo: float, hi: float) -> np.ndarray:
        lo_v = np.percentile(F, lo, axis=0)
        hi_v = np.percentile(F, hi, axis=0)
        return np.clip(F, lo_v, hi_v)

    def balanced_subsample(F: np.ndarray, dom: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        keep = []
        for d in np.unique(dom):
            idx = np.where(dom == d)[0]
            if len(idx) > n:
                idx = rng.choice(idx, size=n, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        return F[keep], dom[keep]

    # ---------------------------
    # 1) LOAD BEFORE EA
    # ---------------------------
    Xa0 = load_subject_trials(subj_a, processed_no_ea)
    Xb0 = load_subject_trials(subj_b, processed_no_ea)

    Fa0 = cov_features(Xa0)
    Fb0 = cov_features(Xb0)

    F0 = np.vstack([Fa0, Fb0])
    dom0 = np.concatenate([np.zeros(len(Fa0), dtype=int), np.ones(len(Fb0), dtype=int)])

    # ---------------------------
    # 2) LOAD AFTER EA
    # ---------------------------
    Xa1 = load_subject_trials(subj_a, processed_w_ea)
    Xb1 = load_subject_trials(subj_b, processed_w_ea)

    Fa1 = cov_features(Xa1)
    Fb1 = cov_features(Xb1)

    F1 = np.vstack([Fa1, Fb1])
    dom1 = np.concatenate([np.zeros(len(Fa1), dtype=int), np.ones(len(Fb1), dtype=int)])

    # ---------------------------
    # 3) CLIP OUTLIERS
    # ---------------------------
    lo, hi = clip_percentiles
    if clip_percentiles is not None:
        F0 = clip_outliers(F0, lo, hi)
        F1 = clip_outliers(F1, lo, hi)

    # ---------------------------
    # 4) BALANCE (SUBSAMPLE)
    # ---------------------------
    if subsample_per_subject is not None:
        F0, dom0 = balanced_subsample(F0, dom0, subsample_per_subject)
        F1, dom1 = balanced_subsample(F1, dom1, subsample_per_subject)

    # ---------------------------
    # 5) STANDARDIZE + PCA
    # Fit on BEFORE, apply to AFTER
    # ---------------------------
    scaler = StandardScaler()
    F0s = scaler.fit_transform(F0)
    F1s = scaler.transform(F1)

    pca = PCA(n_components=2, random_state=random_state)
    Z0 = pca.fit_transform(F0s)
    Z1 = pca.transform(F1s)

    # shared axis limits
    xmin = min(Z0[:, 0].min(), Z1[:, 0].min())
    xmax = max(Z0[:, 0].max(), Z1[:, 0].max())
    ymin = min(Z0[:, 1].min(), Z1[:, 1].min())
    ymax = max(Z0[:, 1].max(), Z1[:, 1].max())

    # ---------------------------
    # 6) PLOT
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, Z, dom, title in [
        (axes[0], Z0, dom0, "Before EA (No EA)"),
        (axes[1], Z1, dom1, "After EA (+EA)"),
    ]:
        ax.scatter(Z[dom == 0, 0], Z[dom == 0, 1], s=10, alpha=0.6, label=pretty_name(subj_a))
        ax.scatter(Z[dom == 1, 0], Z[dom == 1, 1], s=10, alpha=0.6, label=pretty_name(subj_b))

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved EA visualization to: {save_path}")
    print("PCA explained variance ratio (fit on BEFORE):", pca.explained_variance_ratio_)
    

def EEG_data_visualize(subject = "PAT013_processed", trial_idx = 493):
    dataset = MI_EEG_Dataset(subject=subject)

    x, y = dataset[trial_idx]   # x: torch tensor, y: torch scalar
    fs = dataset.fs

    # Convert to numpy
    X = x.detach().cpu().numpy()
    label = int(y.detach().cpu().item())

    n_samples, n_ch = X.shape

    # Time axis
    t = np.arange(n_samples) / fs

    # Channel labels
    ch_names = [
        "FC5", "FC1", "FCz", "FC2", "FC6",
        "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
        "CP5", "CP1", "CP2", "CP6"
    ]


    # Auto spacing
    spacing = 3.0 * np.median(np.std(X, axis=0))
    if not np.isfinite(spacing) or spacing == 0:
        spacing = 1.0

    baselines = spacing * np.arange(n_ch)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))

    for ch in range(n_ch):
        plt.plot(t, X[:, ch] + baselines[ch], linewidth=0.8)

    plt.yticks(baselines, ch_names)
    plt.xlabel("Time (s)")
    plt.title(f"Subject {subject}, EEG Trial {trial_idx}, Label {label} ({n_ch} channels)")
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path("reports") / f"{subject}_trial{trial_idx}_timeseries.png")
    plt.show()

def visualize_frequency_spectrum(subject="PAT013", trial_idx=0, fmax=60):
    dataset = MI_EEG_Dataset(subject=subject)

    print("sampling frequency:", dataset.fs)

    x, y = dataset[trial_idx]   # x: (T,C)
    fs = dataset.fs

    # numpy
    X = x.detach().cpu().numpy()
    label = int(y.detach().cpu().item())

    n_samples, n_ch = X.shape

    # Welch PSD per channel
    nperseg = min(n_samples, int(fs * 2))  # ~2 second window
    f, Pxx = welch(X, fs=fs, axis=0, nperseg=nperseg)  # Pxx: (F,C)

    # keep only up to fmax Hz
    mask = f <= fmax
    f = f[mask]
    Pxx = Pxx[mask, :]

    # log-scale PSD (more readable)
    Pxx_log = np.log10(Pxx + 1e-12)

    # Channel labels:  FC5, FC1, FCz, FC2, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP1, CP2, and CP6.
    ch_names = [
        "FC5", "FC1", "FCz", "FC2", "FC6",
        "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
        "CP5", "CP1", "CP2", "CP6"
    ]

    # Auto spacing like your time plot
    spacing = 3.0 * np.median(np.std(Pxx_log, axis=0))
    if not np.isfinite(spacing) or spacing == 0:
        spacing = 1.0

    baselines = spacing * np.arange(n_ch)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))

    for ch in range(n_ch):
        plt.plot(f, Pxx_log[:, ch] + baselines[ch], linewidth=0.8)

    plt.yticks(baselines, ch_names)
    plt.xlabel("Frequency (Hz)")
    plt.title(f"Subject {subject}, EEG Trial {trial_idx}, Label {label} | Welch PSD ({n_ch} channels)")
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #EEG_data_visualize()
    #visualize_frequency_spectrum()
    #EA_alignment_visualized()
    plot_cross_subject_accuracy_vs_trials(save_path="reports/trials_vs_acc_cross.png")
    plot_within_subject_accuracy_vs_trials(save_path="reports/trials_vs_acc_within.png")