from pathlib import Path
import scipy.io as sio
from torch.utils.data import Dataset
import typer
import matplotlib.pyplot as plt
import numpy as np 
import torch
from collections import defaultdict
import json

from scipy.signal import butter, sosfiltfilt, welch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class MI_EEG_Dataset(Dataset):
    """
    Motor Imagery (MI) EEG dataset (right vs left wrist dorsiflexion imagination)

    Each subject .mat contains:
      - subjectData.fs               (sampling frequency)
      - subjectData.trialsData       cell array of trials (each trial: 1536 x 16)
      - subjectData.trialsLabels     labels for trials (0=right, 1=left)

    Returned tensors:
      x: shape (16, 1536)  (channels, time)
      y: scalar long tensor (0=right, 1=left)
    """

    def __init__(
        self,
        subject: str = "PAT013",
        raw_path: Path = Path("data/Raw_Data"),
        processed_path: Path = Path("data/Processed")
    ):
        """
        Args:
            data_path: Root folder containing Raw_Data/
            subject: Subject file name (e.g. "PAT013" -> PAT013.mat)
        """
        self.subject = subject
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

        self.X = None
        self.y = None
        self.fs = None
        
        # Get the path to the .mat or .pt file
        pt_path = self.processed_path / f"{self.subject}.pt"
        mat_path = self.raw_path / f"{self.subject}.mat"
        
        if mat_path.exists():
            self._load_mat(mat_path)
        elif pt_path.exists():
            self._load_pt(pt_path)
        else:
            raise FileNotFoundError(f"could not find data for subject {self.subject} in {self.raw_path} or {self.processed_path}")
        

    def _load_mat(self, mat_path: Path) -> None:
        hd = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        subject_data = hd["subjectData"]
    
        # Metadata
        self.fs = int(np.asarray(subject_data.fs).squeeze()) if hasattr(subject_data, "fs") else None
        self.subject_id = getattr(subject_data, "subjectId", None)
    
        # trialsData: cell array -> list of arrays (each 1536 x 16)
        trials_data_cell = subject_data.trialsData
        trials_labels = np.asarray(subject_data.trialsLabels).squeeze()
        
        # Convert cell array to a proper (N, 1536, 16) array
        trials_list = []
        for trial in np.atleast_1d(trials_data_cell):
            trial_arr = np.asarray(trial, dtype=np.float32)
            trials_list.append(trial_arr)
        
        self.X = np.stack(trials_list, axis=0)  # (N, 1536, 16)
        self.y = trials_labels.astype(np.int64)  # (N,)
        
    def _load_pt(self, pt_path: Path) -> None:
        data = torch.load(pt_path, map_location="cpu")

        if not all(k in data for k in ("X", "y", "fs")):
            raise KeyError(f"{pt_path} must contain keys: 'X', 'y', 'fs'")

        X = data["X"]
        y = data["y"]

        self.X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        self.y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        self.y = self.y.astype(np.int64)
        self.fs = int(data["fs"])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return int(len(self.X))

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        x = self.X[index]          # (1536, 16)
        y = self.y[index]          # scalar

        # Convert to torch
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def preprocess(self, output_folder: Path, apply_bandpass: bool = True, apply_car: bool = True, apply_z_score: bool = True) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Start with raw data
        X_filt = self.X.copy()
        
        # Apply bandpass filter if specified
        if apply_bandpass:
            # apply the wide bandpass filter
            X_filt = bandpass(X_filt, self.fs, band=(4, 40), order=4)
        
        # Apply CAR (Common Average Reference) (Subtract mean across channels at each time sample.)
        if apply_car:
            X_filt = X_filt - X_filt.mean(axis=2, keepdims=True)

        # Apply z-score normalization per channel
        if apply_z_score:
            mean = X_filt.mean(axis=1, keepdims=True)  # mean over time
            std = X_filt.std(axis=1, keepdims=True) + 1e-12
            X_filt = (X_filt - mean) / std
        
        # Save preprocessed data as .pt
        save_path = output_folder / f"{self.subject}_processed.pt"
        torch.save(
            {
                "X": torch.tensor(X_filt, dtype=torch.float32),
                "y": torch.tensor(self.y, dtype=torch.long),
                "fs": self.fs,
            },
            save_path
        )

def find_best_bandpass_for_all_subjects(data_path: Path = Path("data/Raw_Data"), 
    subjects: list[str] | None = None,
    save_path: Path = Path("data/best_global_bandpass.json"),
) -> None:
    """ 
    Find the overall best Butterworth bandpass filter (band + order)
    across all subjects by averaging CV accuracies. 
    
    Saves the best band+order to a JSON file in data.

    Returns a dict with:
      - best_band
      - best_order
      - mean_acc
      - per_subject_best
      - aggregated_scores
    
    """
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stores list of accuracies per (band, order) across subjects
    aggregated_scores = defaultdict(list)
    
    per_subject_best = {}
    
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        dataset = MI_EEG_Dataset(data_path=data_path, subject=subject)
        best_band, best_order, best_acc, scores = bandpass_sweep(dataset.X, dataset.y, dataset.fs)
        per_subject_best[subject] = {
            "best_band": list(best_band),
            "best_order": int(best_order),
            "best_acc": float(best_acc),
        }
        # Aggregate scores
        for (band, order), acc in scores.items():
            aggregated_scores[(band, order)].append(float(acc))
    
    # Compute global mean score for each (band, order)
    global_mean_scores = {
        (band, order): float(np.mean(accs))
        for (band, order), accs in aggregated_scores.items()
        if len(accs) > 0
    }
    
    # Best overall
    best_key = max(global_mean_scores, key=global_mean_scores.get)
    best_band, best_order = best_key
    best_mean_acc = global_mean_scores[best_key]
    
    print(f"GLOBAL BEST band={best_band}, order={best_order}, mean CV acc={best_mean_acc:.4f}")

    result = {
        "best_band": list(best_band),
        "best_order": int(best_order),
        "mean_acc_across_subjects": float(best_mean_acc),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "per_subject_best": per_subject_best,
        "aggregated_scores_mean": {
            str((list(band), int(order))): float(acc)
            for (band, order), acc in global_mean_scores.items()
        },
    }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return None


def bandpass_sweep(X_raw, y, fs):
    """
    Sweep the bandpass filter parameters through cross-validation.
    
    How it works:
    1) Try several candidate frequency bands (e.g. 8–13 Hz mu, 13–30 Hz beta) and filter orders.
    2) For each band+order, run 5-fold cross-validation so it doesn’t overfit to one train/test split.
    3) In each fold:
       - Bandpass filter the training and test EEG trials.
       - Convert each trial into simple features using log bandpower per channel.
       - Standardize the features (zero mean / unit variance).
       - Train a logistic regression classifier on the training set.
       - Measure accuracy on the test set.
    4) We average the accuracy across folds, and keep the band+order that performs best.
    """

    # Sweep through different orders and cutoff frequencies
    bands = [
        (4, 40),
        (6, 35),
        (6, 30),
        (8, 30),
        (8, 28),
        (8, 24),
        (8, 20),
        (8, 13),   # mu
        (13, 30),  # beta
    ]
    
    # Orders to try
    orders = [2, 4, 6]
    
    # Define StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    #
    best = None
    best_acc = -1
    scores = {}

    for band in bands:
        for order in orders:
            accs = []
            for tr, te in skf.split(X_raw, y):
                Xtr = bandpass(X_raw[tr], fs, band, order)
                Xte = bandpass(X_raw[te], fs, band, order)

                Ftr = log_bandpower(Xtr, fs, band)
                Fte = log_bandpower(Xte, fs, band)

                sc = StandardScaler().fit(Ftr)
                Ftr = sc.transform(Ftr)
                Fte = sc.transform(Fte)

                clf = LogisticRegression(max_iter=2000)
                clf.fit(Ftr, y[tr])
                accs.append(clf.score(Fte, y[te]))

            m = float(np.mean(accs))
            scores[(band, order)] = m

            if m > best_acc:
                best_acc = m
                best = (band, order)
                
    # Return the best band, order, accuracy, and all scores
    best_band, best_order = best
    return best_band, best_order, best_acc, scores


def bandpass(X, fs, band=(8, 30), order=4):
    """ Apply a Butterworth bandpass filter to the data """
    # X: (N, T, C)
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=1)

def log_bandpower(X, fs, band):
    """ Compute the log-bandpower of the signal in the specified frequency band """
    # X: (N, T, C) -> (N, C)
    # How it works:
    # 1) Welch estimates the power spectrum of each trial/channel (how much energy is in each frequency).
    # 2) Select only the frequencies inside the chosen band (e.g. 8–13 Hz for mu rhythm).
    # 3) Integrate the power over that band → total band power for each trial/channel.
    # 4) Take the log of the band power to make values more stable and easier to compare.

    f, Pxx = welch(X, fs=fs, axis=1, nperseg=min(X.shape[1], int(fs * 2)))
    idx = (f >= band[0]) & (f <= band[1])
    bp = np.trapezoid(Pxx[:, idx, :], f[idx], axis=1)
    return np.log(bp + 1e-12)

def preprocess(subject: str = "PAT013") -> None:
    print("Preprocessing data...")
    dataset = MI_EEG_Dataset(subject=subject)
    dataset.preprocess(output_folder=Path("data/Processed"), apply_bandpass=True, apply_car=True, apply_z_score=True)
    #find_best_bandpass_for_all_subjects(
    #    data_path=Path("data/Raw_Data"),
    #    subjects=[
    #        "PAT013", "PAT015", "PAT021_A", "PATID15", "PATID16", "PATID26"
    #    ],
    #    save_path=Path("data/best_global_bandpass.json"),
    #)

if __name__ == "__main__":
    typer.run(preprocess)