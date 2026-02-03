import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .FBCSP import FBCSP
from .rLDA import rLDA
from .data import MI_EEG_Dataset

def save_results(save_path: str, title: str, results_dict: dict):
    """
    Save results to a .txt file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # compute overall mean from subject entries
    subject_accs = list(results_dict.values())
    mean_acc = float(np.mean(subject_accs)) if len(subject_accs) > 0 else float("nan")
    std_acc = float(np.std(subject_accs)) if len(subject_accs) > 0 else float("nan")

    with open(save_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")

        for subj, acc in results_dict.items():
            f.write(f"{subj}: {acc:.4f}\n")

        f.write("\n")
        f.write(f"Overall mean accuracy: {mean_acc:.4f}\n")
        f.write(f"Overall std accuracy : {std_acc:.4f}\n")

    print(f"Saved results to: {save_path}")


def baseline_within(subjects, test_size = 0.2, n_components = 4, lam = 0.1):
    """
    Within-subject baseline: one split per subject
    
    For each subject:
        - split subject trials into train/test once
        - fit CSP on subject-train only
        - train rLDA on CSP features
        - test on subject-test
    """

    results = {}
    for subj in subjects:
        # Load data
        data = MI_EEG_Dataset(subject=subj, processed_path=Path("data/Processed_w_EA"))
        
        # Prepare data
        X = data.X.astype(np.float64)      # (N, T, C)
        y = data.y.astype(int)             # (N,)
        X = X.transpose(0, 2, 1)         # -> (N, C, T) for CSP
        
        # Train/test split INSIDE the subject
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # CSP (fit on train only)
        fbcsp = FBCSP(fs=data.fs, n_components=n_components)
        
        # CSP features
        F_train = fbcsp.fit_transform(X_train, y_train)
        F_test  = fbcsp.transform(X_test)
        
        # rLDA
        clf = rLDA(lam=lam)
        clf.fit(F_train, y_train)
        y_hat = clf.predict(F_test)

        acc = accuracy_score(y_test, y_hat)
        results[subj] = float(acc)
        
        print(f"[WITHIN] {subj}: acc = {acc:.3f}")
        
    overall = float(np.mean(list(results.values())))
    print(f"\n[WITHIN] Mean accuracy across subjects: {overall:.3f}\n")
    return results
        
    
def baseline_cross(subjects, n_components = 4, lam = 0.1):
    """
    Cross-subject baseline: leave-one-subject-out
    
    Repeat 6 times:
        - Train on 5 subjects
        - Test on 1 held-out subject
    """
    results = {}

    # Load all subjects once
    all_X = {}
    all_y = {}
    
    for subj in subjects:
        data = MI_EEG_Dataset(subject=subj, processed_path=Path("data/Processed_w_EA"))
        X = data.X.astype(np.float64)  # (N, T, C)
        y = data.y.astype(int)        # (N,)

        # Convert to CSP shape (N, C, T)
        all_X[subj] = X.transpose(0, 2, 1)
        all_y[subj] = y
    
    for test_subj in subjects:
        # Test set = held-out subject
        X_test = all_X[test_subj]
        y_test = all_y[test_subj]
        
        # Train set = all other subjects
        X_train = np.concatenate([all_X[s] for s in subjects if s != test_subj], axis=0)
        y_train = np.concatenate([all_y[s] for s in subjects if s != test_subj], axis=0)
        
        # CSP (fit on TRAIN subjects only)
        fbcsp = FBCSP(fs=data.fs, n_components=n_components)
        
        # CSP features
        F_train = fbcsp.fit_transform(X_train, y_train)
        F_test  = fbcsp.transform(X_test)
        
        # rLDA
        clf = rLDA(lam=lam)
        clf.fit(F_train, y_train)
        y_hat = clf.predict(F_test)

        acc = accuracy_score(y_test, y_hat)
        results[test_subj] = float(acc)
        
        print(f"[CROSS] Test on {test_subj}: acc = {acc:.3f}")

    overall = float(np.mean(list(results.values())))
    print(f"\n[CROSS] Mean LOSO accuracy: {overall:.3f}\n")
    return results
        
if __name__ == "__main__":
    
    EA = False
    
    subjects = [
        "PAT013_processed", "PAT015_processed", "PAT021_A_processed",
        "PATID15_processed", "PATID16_processed", "PATID26_processed"
    ]
    
    n_comp_within = 6
    lam_within = 0.01

    n_comp_cross = 4
    lam_cross = 0.05

    within_results = baseline_within(
        subjects, test_size=0.2, n_components=n_comp_within, lam=lam_within
    )
    cross_results = baseline_cross(
        subjects, n_components=n_comp_cross, lam=lam_cross
    )

    within_mean = float(np.mean(list(within_results.values())))
    cross_mean = float(np.mean(list(cross_results.values())))

    # ---- Save BEST results ----
    save_results(
        "reports/FBCSP_rLDA_w_bp_w_ea/best_within_results.txt",
        f"FBCSP + rLDA BEST WITHIN (n_components={n_comp_within}, lam={lam_within})",
        within_results
    )
    
    save_results(
        "reports/FBCSP_rLDA_w_bp_w_ea/best_cross_results.txt",
        f"FBCSP + rLDA BEST CROSS (n_components={n_comp_cross}, lam={lam_cross})",
        cross_results
    )
