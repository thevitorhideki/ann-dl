"""Shared pieces of the classification delivery (Bank Marketing, target y = yes).

Everything that more than one step needs lives here: data loading, the EDA split,
the validation split, the network, the training loop with early stopping and the
metrics. Each step script imports from this module, so a number in the report always
comes from the same code path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE = Path(__file__).resolve().parent
DELIVERY = CODE.parent
FIGURES = DELIVERY / "figures"
RESULTS = CODE / "results"
EDA_CODE = DELIVERY.parent / "eda" / "code"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# The EDA pipeline is imported, not copied: same transformations, same columns.
sys.path.insert(0, str(EDA_CODE))
from preprocessing import build_preprocessor  # noqa: E402

DATA_PATH = EDA_CODE / "bank-full.csv"
EXPECTED_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"

SEED = 42
VAL_FRACTION = 0.15  # share of the EDA training set held out for validation

# ---------------------------------------------------------------------------
# Step 0: decided BEFORE any model was trained (see the report, step 0)
# ---------------------------------------------------------------------------
MAIN_METRIC = "pr_auc"  # average precision of the positive class (y = yes)
SUCCESS_RELATIVE_GAIN = 0.05  # network PR-AUC >= 1.05 x best baseline PR-AUC (test)
COST_FALSE_POSITIVE = 1.0  # one call to a client who will not subscribe
COST_FALSE_NEGATIVE = 5.0  # one subscriber who is never called


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load bank-full.csv, check it is the exact file the EDA analysed."""
    checksum = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert checksum == EXPECTED_SHA256, "CSV differs from the one analysed in the EDA"
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop(columns="y")
    y = (df["y"] == "yes").astype(int)
    return X, y


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def make_splits() -> Splits:
    """Same train/test split as the EDA, then validation carved out of train."""
    X, y = load_data()
    # Identical call to the EDA notebook (section 9): 80/20, stratified, seed 42.
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=VAL_FRACTION, random_state=SEED, stratify=y_tr_full
    )
    return Splits(
        X_train, X_val, X_test,
        y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),
    )


def fit_pipeline(splits: Splits):
    """Fit the EDA pipeline on the training part only and transform the three sets."""
    pre = build_preprocessor()
    Xtr = pre.fit_transform(splits.X_train).astype(np.float32)
    Xva = pre.transform(splits.X_val).astype(np.float32)
    Xte = pre.transform(splits.X_test).astype(np.float32)
    return pre, Xtr, Xva, Xte


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully connected network that returns ONE logit per row (no sigmoid).

    The sigmoid lives inside BCEWithLogitsLoss during training and is applied
    explicitly only when probabilities are needed.
    """

    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float = 0.0,
                 batchnorm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers.append(nn.Linear(width, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization: keeps the activation variance stable through ReLUs.
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class Config:
    name: str
    hidden: tuple[int, ...] = (64, 32)
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    batchnorm: bool = False
    pos_weight: float = 1.0  # weight of the positive class inside the loss
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_epoch: int = 0


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def train_network(cfg: Config, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
                  yva: np.ndarray, seed: int = SEED) -> tuple[MLP, History]:
    """Mini-batch training with AdamW and early stopping on the validation loss.

    The best weights (lowest validation loss) are kept in memory and restored at
    the end, so the returned model is the one from `best_epoch`, not the last one.
    """
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], cfg.hidden, cfg.dropout, cfg.batchnorm)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)  # controls the shuffling order
    n = len(Xtr_t)

    hist = History()
    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen)
        running = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        hist.train_loss.append(running / n)

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva_t)
            val_loss = loss_fn(logits_val, yva_t).item()
        hist.val_loss.append(val_loss)
        hist.val_pr_auc.append(
            average_precision_score(yva, torch.sigmoid(logits_val).numpy())
        )

        if val_loss < best_loss - 1e-5:
            best_loss, wait, hist.best_epoch = val_loss, 0, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    hist.stopped_epoch = epoch
    model.load_state_dict(best_state)
    return model, hist


# ---------------------------------------------------------------------------
# Metrics and threshold
# ---------------------------------------------------------------------------
def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return (COST_FALSE_POSITIVE * fp + COST_FALSE_NEGATIVE * fn) / len(y_true)


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold that minimizes the average error cost on VALIDATION."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    costs = [expected_cost(y_val, (p_val >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_yes": float(precision_score(y_true, pred, zero_division=0)),
        "recall_yes": float(recall_score(y_true, pred, zero_division=0)),
        "f1_yes": float(f1_score(y_true, pred, zero_division=0)),
        "precision_no": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "recall_no": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1_no": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "cost": expected_cost(y_true, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def assess_success(mean_pr_auc: float, std_pr_auc: float, baseline_pr_auc: float) -> dict:
    """Require both the relative gain and a gap larger than one sample seed std."""
    target = (1 + SUCCESS_RELATIVE_GAIN) * baseline_pr_auc
    gap = mean_pr_auc - baseline_pr_auc
    relative_gain_met = mean_pr_auc >= target
    gain_exceeds_seed_std = gap > std_pr_auc
    return {
        "criterion_target": target,
        "absolute_gain": gap,
        "relative_gain": mean_pr_auc / baseline_pr_auc - 1,
        "relative_gain_met": bool(relative_gain_met),
        "gain_exceeds_seed_std": bool(gain_exceeds_seed_std),
        "criterion_met": bool(relative_gain_met and gain_exceeds_seed_std),
    }


def save_json(obj, name: str) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())
