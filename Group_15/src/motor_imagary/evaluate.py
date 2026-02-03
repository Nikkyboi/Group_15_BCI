from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .model import ciac_Model
from .train_cross import load_all_subjects_pt, get_preds, evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .CSP import fit_csp, csp_logvar_features
from .rLDA import rLDA
from .data import MI_EEG_Dataset
from .FBCSP import FBCSP
import matplotlib.patheffects as pe


def test_subject(
    held_out: int,
    model_name: str = "ciacnet_v5",
    processed_dir: Path = Path("data/Processed_w_EA"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    batch_size: int = 64,
):
    # ---- device ----
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    # ---- load all subjects ----
    X, y, groups, files = load_all_subjects_pt(processed_dir)
    print(f"Found {len(files)} subjects")

    # ---- subject name ----
    held_out_name = files[held_out].stem.replace("_processed", "")
    print("TEST SUBJECT:", held_out_name)

    # ---- test set only ----
    test_idx = (groups == held_out).nonzero(as_tuple=True)[0].numpy()
    test_ds = TensorDataset(X[test_idx], y[test_idx])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---- load model ----
    model_path = models_dir / f"global_test_{held_out_name}_{model_name}.pth"
    print("Loading model:", model_path)

    model = ciac_Model(n_ch=16, n_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- test accuracy ----
    test_acc = evaluate(model, test_loader, device)
    print(f"Test acc on {held_out_name}: {test_acc:.3f}")

    # ---- predictions ----
    y_true, y_pred = get_preds(model, test_loader, device)

    # ---- plots ----
    reports_dir.mkdir(parents=True, exist_ok=True)

    # confusion matrix
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
    disp.plot()
    plt.title(f"Test Confusion Matrix - {held_out_name}")
    plt.savefig(reports_dir / f"test_confusion_{held_out_name}_{model_name}.png")
    plt.close()

    # true vs pred curve
    correct = (y_pred == y_true).numpy().astype(float)

    window = 25  # smoothness (try 10, 25, 50)
    roll = np.convolve(correct, np.ones(window)/window, mode="valid")

    plt.figure()
    plt.plot(roll)
    plt.ylim(0, 1.05)
    plt.xlabel("Test trial index")
    plt.ylabel(f"Rolling accuracy (window={window})")
    plt.title(f"Rolling Test Accuracy - {held_out_name}")
    plt.savefig(reports_dir / f"test_rolling_acc_{held_out_name}_{model_name}.png")
    plt.close()

    print("Saved plots to:", reports_dir)

    return test_acc


def confusion_matrix_within_subject(
    subject="PATID15_processed",
    test_size=0.2,
    n_components=6,
    lam=0.01,
    save_dir=Path("reports/confusion_matrices"),
    random_state=42,
):
    """
    WITHIN-subject confusion matrix for one subject:
    train/test split inside subject trials.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    data = MI_EEG_Dataset(subject=subject, processed_path=Path("data/Processed_w_EA"))
    X = data.X.astype(np.float64)          # (N, T, C)
    y = data.y.astype(int)                 # (N,)
    X = X.transpose(0, 2, 1)               # (N, C, T)

    # split within subject
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # CSP fit on train only
    W = fit_csp(X_train, y_train, n_components=n_components)
    F_train = csp_logvar_features(X_train, W)
    F_test  = csp_logvar_features(X_test, W)

    # rLDA
    clf = rLDA(lam=lam)
    clf.fit(F_train, y_train)
    y_hat = clf.predict(F_test)

    acc = accuracy_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Right (0)", "Left (1)"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"WITHIN Confusion Matrix ({subject})\nacc={acc:.3f}")
    plt.tight_layout()

    save_path = save_dir / f"confmat_WITHIN_{subject}.png"
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[WITHIN] {subject} acc={acc:.4f}")
    print(f"Saved: {save_path}")
    
    
def confusion_matrix_cross_loso_subject(
    subjects,
    test_subject="PATID15_processed",
    n_components=4,
    lam=0.05,
    save_dir=Path("reports/confusion_matrices"),
):
    """
    CROSS-subject LOSO confusion matrix where the test subject is fixed.
    Train = all other subjects, Test = test_subject.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # load all subjects
    all_X, all_y = {}, {}
    fs_ref = None

    for subj in subjects:
        data = MI_EEG_Dataset(subject=subj, processed_path=Path("data/Processed_w_EA"))
        X = data.X.astype(np.float64)      # (N, T, C)
        y = data.y.astype(int)

        all_X[subj] = X.transpose(0, 2, 1) # (N, C, T)
        all_y[subj] = y

        if fs_ref is None:
            fs_ref = data.fs

    if test_subject not in subjects:
        raise ValueError(f"test_subject='{test_subject}' must be in subjects list")

    # test set
    X_test = all_X[test_subject]
    y_test = all_y[test_subject]

    # train set
    X_train = np.concatenate([all_X[s] for s in subjects if s != test_subject], axis=0)
    y_train = np.concatenate([all_y[s] for s in subjects if s != test_subject], axis=0)

    # CSP on train only
    W = fit_csp(X_train, y_train, n_components=n_components)

    # features
    F_train = csp_logvar_features(X_train, W)
    F_test  = csp_logvar_features(X_test, W)

    # rLDA
    clf = rLDA(lam=lam)
    clf.fit(F_train, y_train)
    y_hat = clf.predict(F_test)

    acc = accuracy_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Right (0)", "Left (1)"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"CROSS (LOSO) Confusion Matrix (test={test_subject})\nacc={acc:.3f}")
    plt.tight_layout()

    save_path = save_dir / f"confmat_CROSS_test_{test_subject}.png"
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[CROSS] test={test_subject} acc={acc:.4f}")
    print(f"Saved: {save_path}")


def full_confusion_within_FBCSP_rLDA(
    subjects,
    test_size=0.2,
    n_components=6,
    lam=0.05,
    random_state=42,
    normalize_plot=False,
    save_path=None,
    bands=None,              # optional: pass your own filterbank
    fb_order=4               # optional: butterworth order
):
    """
    Aggregated WITHIN-subject confusion matrix:
    - For each subject: split train/test once
    - Fit FBCSP+rLDA on train, predict test
    - Sum confusion matrices over all subjects
    """

    C_total = np.zeros((2, 2), dtype=int)
    y_all_true = []
    y_all_pred = []

    for subj in subjects:
        ds = MI_EEG_Dataset(subject=subj, processed_path=Path("data/Processed"))

        X = ds.X.astype(np.float64)          # (N, T, C)
        y = ds.y.astype(int)                 # (N,)
        X = X.transpose(0, 2, 1)             # -> (N, C, T)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # ---- FBCSP features ----
        fbcsp = FBCSP(fs=ds.fs, bands=bands, n_components=n_components, order=fb_order)
        F_train = fbcsp.fit_transform(X_train, y_train)
        F_test  = fbcsp.transform(X_test)

        # ---- rLDA ----
        clf = rLDA(lam=lam)
        clf.fit(F_train, y_train)
        y_hat = clf.predict(F_test)

        # accumulate
        C = confusion_matrix(y_test, y_hat, labels=[0, 1])
        C_total += C

        y_all_true.append(y_test)
        y_all_pred.append(y_hat)

        acc = accuracy_score(y_test, y_hat)
        print(f"[WITHIN] {subj}: acc={acc:.3f}")

    y_all_true = np.concatenate(y_all_true)
    y_all_pred = np.concatenate(y_all_pred)

    acc_total = accuracy_score(y_all_true, y_all_pred)

    print("\n[WITHIN] Aggregated accuracy:", round(acc_total, 4))
    print("[WITHIN] Aggregated confusion matrix:\n", C_total)

    # ---- visualization ----
    plot_confusion_matrix(
        C_total,
        title=f"WITHIN Aggregated Confusion Matrix (FBCSP+rLDA)\nacc={acc_total:.3f}",
        normalize=normalize_plot,
        save_path=save_path
    )

    return C_total, acc_total


def full_confusion_cross_FBCSP_rLDA(
    subjects,
    n_components=4,
    lam=0.1,
    normalize_plot=False,
    save_path=None,
    bands=None,
    fb_order=4
):
    """
    Aggregated CROSS-subject confusion matrix (LOSO):
    - For each held-out subject: train on the other subjects
    - Predict on the held-out subject
    - Sum confusion matrices over all LOSO folds
    """

    # Load all subjects once
    all_X = {}
    all_y = {}
    fs_ref = None

    for subj in subjects:
        ds = MI_EEG_Dataset(subject=subj, processed_path=Path("data/Processed"))
        X = ds.X.astype(np.float64).transpose(0, 2, 1)  # (N,C,T)
        y = ds.y.astype(int)

        all_X[subj] = X
        all_y[subj] = y
        fs_ref = ds.fs if fs_ref is None else fs_ref

    C_total = np.zeros((2, 2), dtype=int)
    y_all_true = []
    y_all_pred = []

    for test_subj in subjects:
        X_test = all_X[test_subj]
        y_test = all_y[test_subj]

        X_train = np.concatenate([all_X[s] for s in subjects if s != test_subj], axis=0)
        y_train = np.concatenate([all_y[s] for s in subjects if s != test_subj], axis=0)

        # ---- FBCSP features ----
        fbcsp = FBCSP(fs=fs_ref, bands=bands, n_components=n_components, order=fb_order)
        F_train = fbcsp.fit_transform(X_train, y_train)
        F_test  = fbcsp.transform(X_test)

        # ---- rLDA ----
        clf = rLDA(lam=lam)
        clf.fit(F_train, y_train)
        y_hat = clf.predict(F_test)

        C = confusion_matrix(y_test, y_hat, labels=[0, 1])
        C_total += C

        y_all_true.append(y_test)
        y_all_pred.append(y_hat)

        acc = accuracy_score(y_test, y_hat)
        print(f"[CROSS] test={test_subj}: acc={acc:.3f}")

    y_all_true = np.concatenate(y_all_true)
    y_all_pred = np.concatenate(y_all_pred)

    acc_total = accuracy_score(y_all_true, y_all_pred)

    print("\n[CROSS] Aggregated accuracy:", round(acc_total, 4))
    print("[CROSS] Aggregated confusion matrix:\n", C_total)

    # ---- visualization ----
    plot_confusion_matrix(
        C_total,
        title=f"CROSS (LOSO) Aggregated Confusion Matrix (FBCSP+rLDA)\nacc={acc_total:.3f}",
        normalize=normalize_plot,
        save_path=save_path
    )

    return C_total, acc_total


def plot_confusion_matrix(
    C,
    title="Confusion Matrix",
    class_names=("Right (0)", "Left (1)"),
    normalize=False,
    cmap="viridis",
    save_path=None,
):
    """
    Plot a 2x2 confusion matrix with counts (and optional row-normalized percentages).

    C: np.ndarray shape (2,2)
    normalize:
        False -> show counts
        True  -> show row-normalized % (each row sums to 100%)
    """
    C = np.array(C, dtype=float)

    if normalize:
        row_sums = C.sum(axis=1, keepdims=True) + 1e-12
        C_plot = C / row_sums
    else:
        C_plot = C

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(C_plot, cmap=cmap)

    # Build a normalization object so we can map values -> colors consistently
    norm = im.norm
    cmap_obj = im.cmap

    # text annotations with adaptive color + outline for readability
    for i in range(2):
        for j in range(2):
            if normalize:
                text = f"{C[i, j]:.0f}\n({100*C_plot[i,j]:.1f}%)"
            else:
                text = f"{int(C[i, j])}"

            val = C_plot[i, j]
            r, g, b, _ = cmap_obj(norm(val))

            # Perceived luminance (0=dark, 1=light)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            # Choose text color based on background brightness
            text_color = "white" if luminance < 0.5 else "black"

            # Outline color opposite of text for extra contrast
            outline_color = "black" if text_color == "white" else "white"

            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=12,
                color=text_color,
                path_effects=[pe.withStroke(linewidth=3, foreground=outline_color)]
            )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved confusion matrix plot to: {save_path}")

    plt.show()

if __name__ == "__main__":
    subjects = [
        "PAT013_processed",
        "PAT015_processed",
        "PAT021_A_processed",
        "PATID15_processed",
        "PATID16_processed",
        "PATID26_processed",
    ]

    confusion_matrix_within_subject(
        subject="PATID15_processed",
        test_size=0.2,
        n_components=6,
        lam=0.01
    )

    confusion_matrix_cross_loso_subject(
        subjects=subjects,
        test_subject="PATID15_processed",
        n_components=4,
        lam=0.05
    )
    """
    
    subjects = [
    "PAT013_processed", "PAT015_processed", "PAT021_A_processed",
    "PATID15_processed", "PATID16_processed", "PATID26_processed"
]

    # WITHIN (best within for FBCSP+rLDA)
full_confusion_within_FBCSP_rLDA(
    subjects,
    test_size=0.2,
    n_components=6,
    lam=0.05,
    normalize_plot=False,
    save_path="reports/confmat_WITHIN_aggregated_FBCSP_rLDA.png"
)

# CROSS (best cross for FBCSP+rLDA)
full_confusion_cross_FBCSP_rLDA(
    subjects,
    n_components=4,
    lam=0.1,
    normalize_plot=False,
    save_path="reports/confmat_CROSS_aggregated_FBCSP_rLDA.png"
)
"""