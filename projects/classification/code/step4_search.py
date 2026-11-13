"""Step 4: regularization techniques and hyperparameter search (VALIDATION only).

Part A compares the base network without regularization against weight decay,
dropout and BatchNorm, keeping architecture and learning rate fixed.
Part B searches 16 configurations and sorts them by validation PR-AUC.

Run from the repository root:
    python docs/projects/classification/code/step4_search.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import (FIGURES, RESULTS, SEED, Config, choose_threshold, count_parameters,
                    evaluate, fit_pipeline, make_splits, predict_proba, save_json,
                    train_network)

REGULARIZATION = [
    Config("Sem regularização", hidden=(64, 32)),
    Config("Weight decay 0.05", hidden=(64, 32), weight_decay=0.05),
    Config("Dropout 0.3", hidden=(64, 32), dropout=0.3),
    Config("BatchNorm", hidden=(64, 32), batchnorm=True),
    Config("Dropout 0.3 + weight decay 0.05", hidden=(64, 32), dropout=0.3, weight_decay=0.05),
]


def search_space(pos_ratio: float) -> list[Config]:
    configs = []
    for hidden in [(32,), (64, 32), (128, 64), (128, 64, 32)]:
        tag = "-".join(map(str, hidden))
        for lr in [1e-3, 3e-4]:
            configs.append(Config(f"{tag} lr={lr:g} drop=0.3 wd=0.05", hidden=hidden, lr=lr,
                                  dropout=0.3, weight_decay=0.05))
        configs.append(Config(f"{tag} lr=0.001 drop=0.1 wd=0.01", hidden=hidden, lr=1e-3,
                              dropout=0.1, weight_decay=0.01))
        configs.append(Config(f"{tag} lr=0.001 drop=0.3 wd=0.05 pos_weight", hidden=hidden,
                              lr=1e-3, dropout=0.3, weight_decay=0.05,
                              pos_weight=round(pos_ratio, 2)))
    return configs


def run(cfg: Config, Xtr, ytr, Xva, yva) -> tuple[dict, dict]:
    model, hist = train_network(cfg, Xtr, ytr, Xva, yva, seed=SEED)
    p_val = predict_proba(model, Xva)
    p_tr = predict_proba(model, Xtr)
    t = choose_threshold(yva, p_val)
    m_val = evaluate(yva, p_val, t)
    m_tr = evaluate(ytr, p_tr, t)
    row = {
        "config": cfg.name, "hidden": "-".join(map(str, cfg.hidden)), "lr": cfg.lr,
        "dropout": cfg.dropout, "weight_decay": cfg.weight_decay, "batchnorm": cfg.batchnorm,
        "pos_weight": cfg.pos_weight, "params": count_parameters(model),
        "best_epoch": hist.best_epoch, "stopped_epoch": hist.stopped_epoch,
        "train_pr_auc": m_tr["pr_auc"], "val_pr_auc": m_val["pr_auc"],
        "val_roc_auc": m_val["roc_auc"], "val_f1_yes": m_val["f1_yes"],
        "val_threshold": t, "val_cost": m_val["cost"],
        "val_loss_best": hist.val_loss[hist.best_epoch - 1],
    }
    return row, hist.__dict__


def main() -> None:
    splits = make_splits()
    _, Xtr, Xva, _ = fit_pipeline(splits)
    ytr, yva = splits.y_train, splits.y_val
    pos_ratio = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"negative/positive ratio in train: {pos_ratio:.2f}")

    # ---------------- Part A: regularization techniques ----------------
    reg_rows, reg_hist = [], {}
    for cfg in REGULARIZATION:
        row, hist = run(cfg, Xtr, ytr, Xva, yva)
        reg_rows.append(row)
        reg_hist[cfg.name] = hist
        print(f"[A] {cfg.name:34s} best ep {row['best_epoch']:3d} | train PR-AUC "
              f"{row['train_pr_auc']:.4f} | val PR-AUC {row['val_pr_auc']:.4f} | "
              f"gap {row['train_pr_auc'] - row['val_pr_auc']:+.4f}")
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(RESULTS / "step4_regularization.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    colors = ["#222222", "#466B91", "#D27632", "#5E9C5A", "#8A5BA6"]
    for (name, hist), color in zip(reg_hist.items(), colors):
        ep = range(1, len(hist["val_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], color=color, label=name)
        axes[1].plot(ep, hist["val_loss"], color=color, label=name)
        axes[1].scatter(hist["best_epoch"], hist["val_loss"][hist["best_epoch"] - 1],
                        color=color, s=25, zorder=3)
    axes[0].set(title="Treino", xlabel="Época", ylabel="BCE (nats por linha)")
    axes[1].set(title="Validação (ponto = melhor época)", xlabel="Época")
    axes[0].set_ylim(0.24, 0.36)
    axes[1].legend(fontsize=8)
    fig.suptitle("Figura 2 — Regularização: rede 64-32, lr 0.001, mesma seed")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-regularizacao.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Part B: hyperparameter search ----------------
    rows = []
    for cfg in search_space(pos_ratio):
        row, _ = run(cfg, Xtr, ytr, Xva, yva)
        rows.append(row)
        print(f"[B] {cfg.name:40s} params {row['params']:6d} | best ep {row['best_epoch']:3d} | "
              f"val PR-AUC {row['val_pr_auc']:.4f} | ROC {row['val_roc_auc']:.4f}")
    df = pd.DataFrame(rows).sort_values(["val_pr_auc", "params"], ascending=[False, True])
    df.to_csv(RESULTS / "step4_search.csv", index=False)
    best = df.iloc[0].to_dict()
    save_json({"pos_ratio": float(pos_ratio), "best": best,
               "n_configs": len(df)}, "step4_best.json")
    print("\nBEST:", best["config"], f"val PR-AUC {best['val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
