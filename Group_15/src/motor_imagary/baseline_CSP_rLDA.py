import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .CSP import fit_csp, csp_logvar_features
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
        data = MI_EEG_Dataset(subject=subj)
        
        # Prepare data
        X = data.X.astype(np.float64)      # (N, T, C)
        y = data.y.astype(int)             # (N,)
        X = X.transpose(0, 2, 1)         # -> (N, C, T) for CSP
        
        # Train/test split INSIDE the subject
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # CSP (fit on train only)
        W = fit_csp(X_train, y_train, n_components=n_components)
        
        # CSP features
        F_train = csp_logvar_features(X_train, W)
        F_test  = csp_logvar_features(X_test, W)
        
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
        data = MI_EEG_Dataset(subject=subj)
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
        W = fit_csp(X_train, y_train, n_components=n_components)
        
        # CSP features
        F_train = csp_logvar_features(X_train, W)
        F_test  = csp_logvar_features(X_test, W)
        
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
    subjects = [
        "PAT013_processed", "PAT015_processed", "PAT021_A_processed", "PATID15_processed", "PATID16_processed", "PATID26_processed"
    ]
    within_results = baseline_within(subjects, test_size=0.2, n_components=4, lam=0.1)
    cross_results = baseline_cross(subjects, n_components=4, lam=0.1)
    
    # Save to txt files
    save_results(
        "reports/baseline_CSP_rLDA/within_results.txt",
        "CSP + rLDA Baseline (WITHIN-subject)",
        within_results
    )

    save_results(
        "reports/baseline_CSP_rLDA/cross_results.txt",
        "CSP + rLDA Baseline (CROSS-subject LOSO)",
        cross_results
    )