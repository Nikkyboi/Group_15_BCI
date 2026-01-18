from pathlib import Path
import scipy.io as sio
from torch.utils.data import Dataset
import typer
import matplotlib.pyplot as plt
import numpy as np 
import torch
from .data import MI_EEG_Dataset

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
    ch_names = [f"Channel {i+1}" for i in range(n_ch)]

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
    plt.show()


if __name__ == "__main__":
    EEG_data_visualize()