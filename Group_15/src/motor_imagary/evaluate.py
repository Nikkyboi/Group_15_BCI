from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .model import ciac_Model
from .train import load_all_subjects_pt, get_preds, evaluate
import numpy as np

def test_subject(
    held_out: int,
    model_name: str = "ciacnet_v5",
    processed_dir: Path = Path("data/Processed"),
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

if __name__ == "__main__":
    test_subject(held_out=0)  # PAT013 (index 0)