from pathlib import Path
import scipy.io as sio
from torch.utils.data import Dataset
import typer
import matplotlib.pyplot as plt
import numpy as np 
import torch
from collections import defaultdict
import json

from scipy.signal import iirnotch, filtfilt, detrend
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
        subject: str = "PAT021_A_processed",
        raw_path: Path = Path("data/Raw_data"),
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

    def preprocess(self, output_folder: Path, apply_bandpass: bool = True, apply_car: bool = True, apply_ea: bool = True, apply_z_score: bool = False) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Start with raw data
        X_filt = self.X.copy()

        # TESTING: cropping inside the 6s MI window (e.g. keep 1s–5s)
        start = int(0.5 * self.fs)
        end   = int(5.0 * self.fs)
        X_filt = X_filt[:, start:end, :]
        
        # Remove DC offset
        X_filt = X_filt - X_filt.mean(axis=1, keepdims=True)
        #X_filt = detrend(X_filt, axis=1, type="linear") # detrend each trial
        
        # Remove power line noise with notch filter at 50 Hz
        b, a = iirnotch(w0=50.0, Q=30.0, fs=self.fs)
        X_filt = filtfilt(b, a, X_filt, axis=1)
        
        # Apply bandpass filter if specified
        if apply_bandpass:
            # apply the wide bandpass filter
            X_filt = bandpass(X_filt, self.fs, band=(8, 30), order=4)
        
        # Apply CAR (Common Average Reference) (Subtract mean across channels at each time sample.)
        if apply_car:
            X_filt = X_filt - X_filt.mean(axis=2, keepdims=True)

        # (for testing) EA alignment
        if apply_ea:
            X_filt = ea_align_subject(X_filt)
        
        # Z-score per channel (normalize over time)
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
        
# Euclidean Alignment
def ea_align_subject(X):
    # X: (N, T, C)
    # compute mean covariance across trials
    C = []
    for i in range(X.shape[0]):
        Xi = X[i]  # (T,C)
        Ci = np.cov(Xi.T)  # (C,C)
        C.append(Ci)
    R = np.mean(C, axis=0)

    # R^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(R)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
    R_inv_sqrt = eigvecs @ D_inv_sqrt @ eigvecs.T

    # apply to all trials: (T,C) -> (T,C)
    return np.einsum("ij,ntj->nti", R_inv_sqrt, X)

def bandpass(X, fs, band=(8, 30), order=4):
    """ Apply a Butterworth bandpass filter to the data """
    # X: (N, T, C)
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=1)

def preprocess(subject: str = "PAT013") -> None:
    print("Preprocessing data...")
    subjects = [
        "PAT013", "PAT015", "PAT021_A", "PATID15", "PATID16", "PATID26"
    ]
    for subject in subjects:
        dataset = MI_EEG_Dataset(subject=subject)
        dataset.preprocess(output_folder=Path("data/Processed"), apply_bandpass=True, apply_car=True, apply_ea=False, apply_z_score=False)

if __name__ == "__main__":
    typer.run(preprocess)