from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .model import ciac_Model
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_all_subjects_pt(processed_folder: Path):
    files = sorted(processed_folder.glob("*_processed.pt"))
    X_all, y_all, g_all = [], [], []

    for subj_id, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        X = d["X"]  # (N, T, C)
        y = d["y"]  # (N,)

        # (N,T,C) -> (N,C,T)
        X = X.permute(0, 2, 1).contiguous()

        X_all.append(X)
        y_all.append(y)

        # group label for each trial = subject id
        g_all.append(torch.full((len(y),), subj_id, dtype=torch.long))

    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    g_all = torch.cat(g_all, dim=0)

    return X_all, y_all, g_all, files

def get_preds(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).argmax(dim=1).cpu()
            y_true.append(y.cpu())
            y_pred.append(pred)
    return torch.cat(y_true), torch.cat(y_pred)

def train_one_model(model, train_loader, val_loader, device, epochs=200, lr=1e-3, logger=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, verbose=True
    )
    # added class weights to handle class imbalance
    #weights = torch.tensor([1.0, 1.3], device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None

    train_losses = []
    val_losses = []
    
    print(f"Initial LR: {opt.param_groups[0]['lr']:.2e}")
    
    for epoch in range(1, epochs + 1):

        # -------- TRAIN --------
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            running_train_loss += loss.item() * X.size(0)
            n_train += X.size(0)

        train_loss = running_train_loss / n_train
        train_losses.append(train_loss)

        # -------- VAL --------
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        n_val = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)

                loss = loss_fn(out, y)
                running_val_loss += loss.item() * X.size(0)
                n_val += X.size(0)

                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        val_loss = running_val_loss / n_val
        val_losses.append(val_loss)

        val_acc = correct / total

        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        if logger:
            logger.info(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | lr={current_lr:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
            )

        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val, train_losses, val_losses




def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

if __name__ == "__main__":
    """
    
    This is an example of how the training setup can be used for EEG classification.
    
    One model:
    - CIACNet (CNN)
    
    For EEG dataset preprocessing, see data.py:
    - preprocess(): applies bandpass filter, z-score normalization
    - apply_car: apply common average referencing
    - Train the model on multiple subjects' data
    
    """
    # ----------------------------
    # Set device (Use GPU if available)
    # dataset = MyDataset(...)
    device = None
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    use_logger = False
    logger = None
    if use_logger:
        # Logging
        logger = logging.getLogger("eeg_training")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
        logger.info(f"Device: {device}")
    
    # --------------------------
    # load data
    X, y, groups, files = load_all_subjects_pt(Path("data/Processed"))
    
    if use_logger:
        logger.info(f"Found {len(files)} subjects")
    else:
        print(f"Found {len(files)} subjects")
    
    # Pick one subject to hold out for testing
    subject_id = 0  # PAT013
    subject_name = files[subject_id].stem.replace("_processed", "")
    print("SUBJECT:", subject_name)

    
    idx = (groups == subject_id).nonzero(as_tuple=True)[0].numpy()

    # split into train / val / test within the same subject
    train_idx, tmp_idx = train_test_split(
        idx,
        test_size=0.30,  # 70% train, 30% remaining
        stratify=y[idx].numpy(),
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=0.50,  # 15% val, 15% test
        stratify=y[tmp_idx].numpy(),
        random_state=42
    )
    
    # DataLoaders
    batch_size = 64
    
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx], y[val_idx])
    test_ds  = TensorDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model path
    model_name = "ciacnet_v5_within"
    Path("models/subject_dependent").mkdir(parents=True, exist_ok=True)
    model_path = Path("models/subject_dependent") / f"{subject_name}_{model_name}.pth"

    # Model hyperparameters
    n_in = 16
    n_out = 2
    epochs = 250
    lr = 1e-3

    # Train model
    model = ciac_Model(n_ch=16, n_classes=2)

    best_val, train_loss_total, val_loss_total = train_one_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, logger=None
    )

    test_acc = evaluate(model, test_loader, device)

    torch.save(model.state_dict(), model_path)

    if use_logger:
        logger.info(f"Best val acc: {best_val:.3f}")
        logger.info(f"Test acc:     {test_acc:.3f}")
        logger.info(f"Saved model: {model_path}")
    else:
        print(f"Best val acc: {best_val:.3f}")
        print(f"Test acc:     {test_acc:.3f}")
        print(f"Saved model: {model_path}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.plot(train_loss_total, label="Train Loss")
    plt.plot(val_loss_total, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss for " + model_name)
    plt.savefig(reports_dir / f"training_validation_loss_{subject_name}_{model_name}.png")
    plt.close()

    # Confusion matrix plot (prediction graph)
    y_true, y_pred = get_preds(model, test_loader, device)

    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {subject_name}")
    plt.savefig(reports_dir / f"confusion_matrix_{subject_name}_{model_name}.png")
    plt.close()