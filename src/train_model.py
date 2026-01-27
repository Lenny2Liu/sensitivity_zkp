
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader






ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult_dataframe(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Adult CSV not found at {csv_path}")

    df = pd.read_csv(
        csv_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
    )
    target_col = "income"
    return df, target_col


def preprocess_adult(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.replace("?", np.nan).dropna(axis=0).copy()

    y_raw = df[target_col].astype(str).str.strip()
    mapping = {">50K": 1, ">50K.": 1, "<=50K": 0, "<=50K.": 0}
    if not set(y_raw.unique()).issubset(mapping.keys()):
        raise ValueError(f"Unexpected income labels: {sorted(y_raw.unique())}")
    y = y_raw.replace(mapping).astype(int)

    df_features = df.drop(columns=[target_col])

    numeric_cols = df_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    X_num = df_features[numeric_cols].astype(float).values
    X_cat = df_features[categorical_cols].astype(str).values

    if X_num.size > 0:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = np.zeros((len(df_features), 0), dtype=np.float32)

    if X_cat.size > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_encoded = ohe.fit_transform(X_cat)
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    else:
        X_cat_encoded = np.zeros((len(df_features), 0), dtype=np.float32)
        ohe_feature_names = []

    X = np.concatenate([X_num_scaled, X_cat_encoded], axis=1).astype(np.float32)
    feature_names = numeric_cols + ohe_feature_names
    y_arr = y.to_numpy(dtype=np.int64)

    return X, y_arr, feature_names






class AdultConvDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, image_size: int = 32):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.image_size = image_size
        self.num_pixels = image_size * image_size
        self.dim = X.shape[1]
        if self.dim > self.num_pixels:
            raise ValueError(
                f"Feature dim {self.dim} is larger than {self.num_pixels} pixels. "
                "You'd need a bigger image or feature selection."
            )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        v = self.X[idx]  
        img = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        flat = img.reshape(-1)  
        flat[: self.dim] = v
        img = flat.reshape(1, self.image_size, self.image_size)
        return torch.from_numpy(img), int(self.y[idx])


def make_dataloaders_conv(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    train_ds = AdultConvDataset(X_train, y_train, image_size=32)
    val_ds = AdultConvDataset(X_val, y_val, image_size=32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader






# class AdultLeNetConv(nn.Module):
#     """
#     Conv architecture matching CNN.cpp LENET branch:

#       conv1: (1 -> 4) k=5, avgpool
#       conv2: (4 -> 16) k=5, avgpool
#       conv3: (16 -> 128) k=5
#       flatten -> 128 -> 32 -> 16

#     All layers without bias to match add_filter/add_dense.
#     """

#     def __init__(self, in_channels: int = 1, num_classes: int = 16):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5, bias=False)
#         self.conv2 = nn.Conv2d(4, 16, kernel_size=5, bias=False)
#         self.conv3 = nn.Conv2d(16, 128, kernel_size=5, bias=False)

#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU(inplace=True)

#         self.fc1 = nn.Linear(128, 32, bias=False)
#         self.fc2 = nn.Linear(32, num_classes, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         x = self.relu(self.conv1(x))  
#         x = self.pool(x)              
#         x = self.relu(self.conv2(x))  
#         x = self.pool(x)              
#         x = self.relu(self.conv3(x))  
#         x = torch.flatten(x, 1)       
#         x = self.relu(self.fc1(x))    
#         x = self.fc2(x)               
#         return x


class AdultLeNetConv(nn.Module):
    """
    Wider LeNet-style CNN for Adult dataset.

    Architecture (32x32 input):

      conv1:  (1   -> 16)  k=5, no bias, ReLU, 2x2 avgpool
      conv2:  (16  -> 64)  k=5, no bias, ReLU, 2x2 avgpool
      conv3:  (64  -> 384) k=5, no bias, ReLU

      After conv/pool:
        32x32
        -> conv1(5x5)      -> 16x28x28
        -> pool(2x2, s=2)  -> 16x14x14
        -> conv2(5x5)      -> 64x10x10
        -> pool(2x2, s=2)  -> 64x5x5
        -> conv3(5x5)      -> 384x1x1
        -> flatten         -> 384

      fc1: 384 -> 384 (no bias, ReLU)
      fc2: 384 -> num_classes (no bias)

    Total parameters (no biases):
      conv1: 16 * 1   * 5 * 5   =        400
      conv2: 64 * 16  * 5 * 5   =     25,600
      conv3: 384 * 64 * 5 * 5   =    614,400
      fc1:   384 * 384          =    147,456
      fc2:   384 * num_classes  = 6,144 for num_classes=16

      Sum for num_classes = 16:
        400 + 25,600 + 614,400 + 147,456 + 6,144 = 794,000 parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 16):
        super().__init__()

        # Convolutional part (still "LeNet style", just wider)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, bias=False)
        self.conv3 = nn.Conv2d(64, 384, kernel_size=5, bias=False)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # Fully-connected part
        self.fc1 = nn.Linear(384, 384, bias=False)
        self.fc2 = nn.Linear(384, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 32, 32)
        x = self.relu(self.conv1(x))   # (N, 16, 28, 28)
        x = self.pool(x)               # (N, 16, 14, 14)
        x = self.relu(self.conv2(x))   # (N, 64, 10, 10)
        x = self.pool(x)               # (N, 64, 5, 5)
        x = self.relu(self.conv3(x))   # (N, 384, 1, 1)

        x = torch.flatten(x, 1)        # (N, 384)
        x = self.relu(self.fc1(x))     # (N, 384)
        x = self.fc2(x)                # (N, num_classes)
        return x




def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def export_for_cpp(model: AdultLeNetConv, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()

    weights = {}
    conv_id = 0
    fc_id = 0
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if "conv" in name and "weight" in name:
            key = f"conv{conv_id}_weight"
            conv_id += 1
        elif "fc" in name and "weight" in name:
            key = f"fc{fc_id}_weight"
            fc_id += 1
        else:
            continue
        weights[key] = arr
        txt_path = out_dir / f"{key}.txt"
        np.savetxt(txt_path, arr.reshape(1, -1))
        print(f"Exported {key} shape={arr.shape} -> {txt_path}")

    
    npz_path = out_dir / "adult_lenet_weights.npz"
    np.savez(npz_path, **weights)
    print(f"Saved combined weights to {npz_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]  
    default_csv = project_root / "datasets" / "fairness" / "adult" / "raw" / "adult.csv"
    default_out = project_root / "models" / "adult_conv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(default_csv))
    parser.add_argument("--out-dir", type=str, default=str(default_out))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading Adult from {csv_path}")

    df, target_col = load_adult_dataframe(csv_path)
    X, y, feature_names = preprocess_adult(df, target_col)
    print(f"X shape = {X.shape}, y shape = {y.shape}, n_features = {len(feature_names)}")

    train_loader, val_loader = make_dataloaders_conv(X, y, batch_size=args.batch_size)

    model = AdultLeNetConv(in_channels=1, num_classes=16)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val_acc = {best_val_acc:.4f}")

    
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "adult_lenet_state_dict.pt")

    
    export_for_cpp(model, out_dir)


if __name__ == "__main__":
    main()
