"""Steps 2 and 3: the base network and its training curves (validation only).

Run from the repository root:
    python docs/projects/classification/code/step2_3_network.py
"""

import time

import matplotlib.pyplot as plt

from common import (FIGURES, SEED, Config, choose_threshold, count_parameters, evaluate,
                    fit_pipeline, make_splits, predict_proba, save_json, train_network)

BASE = Config(name="base", hidden=(64, 32), lr=1e-3, batch_size=256,
              max_epochs=200, patience=15)


def main() -> None:
    splits = make_splits()
    _, Xtr, Xva, _ = fit_pipeline(splits)

    t0 = time.time()
    model, hist = train_network(BASE, Xtr, splits.y_train, Xva, splits.y_val, seed=SEED)
    elapsed = time.time() - t0
    n_params = count_parameters(model)

    p_val = predict_proba(model, Xva)
    t = choose_threshold(splits.y_val, p_val)
    m_val = evaluate(splits.y_val, p_val, t)
    m_train = evaluate(splits.y_train, predict_proba(model, Xtr), t)

    print(model)
    print(f"parameters: {n_params:,} | training rows: {len(Xtr):,} "
          f"| rows per parameter: {len(Xtr) / n_params:.2f}")
    print(f"best epoch {hist.best_epoch} | stopped at {hist.stopped_epoch} | {elapsed:.1f}s")
    print(f"train loss at best epoch {hist.train_loss[hist.best_epoch - 1]:.4f} | "
          f"val loss {hist.val_loss[hist.best_epoch - 1]:.4f}")
    print(f"min train loss {min(hist.train_loss):.4f} (epoch "
          f"{hist.train_loss.index(min(hist.train_loss)) + 1}) | last val loss {hist.val_loss[-1]:.4f}")
    print(f"VAL  PR-AUC {m_val['pr_auc']:.4f} | ROC-AUC {m_val['roc_auc']:.4f} | t={t:.2f} "
          f"F1(yes) {m_val['f1_yes']:.4f}")
    print(f"TRAIN PR-AUC {m_train['pr_auc']:.4f} | ROC-AUC {m_train['roc_auc']:.4f}")

    save_json({"config": BASE.to_dict(), "n_params": n_params, "n_train": len(Xtr),
               "best_epoch": hist.best_epoch, "stopped_epoch": hist.stopped_epoch,
               "seconds": elapsed, "history": hist.__dict__,
               "val": m_val, "train": m_train}, "step2_3_network.json")

    epochs = range(1, len(hist.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(epochs, hist.train_loss, label="Treino", color="#466B91")
    axes[0].plot(epochs, hist.val_loss, label="Validação", color="#D27632")
    axes[0].axvline(hist.best_epoch, color="gray", ls="--",
                    label=f"Melhor época ({hist.best_epoch})")
    axes[0].set(title="Perda BCE por época", xlabel="Época", ylabel="BCE (nats por linha)")
    axes[0].legend()
    axes[1].plot(epochs, hist.val_pr_auc, color="#D27632", label="Validação")
    axes[1].axvline(hist.best_epoch, color="gray", ls="--", label="Pesos restaurados")
    axes[1].set(title="PR-AUC na validação por época", xlabel="Época", ylabel="PR-AUC (yes)")
    axes[1].legend()
    fig.suptitle("Figura 1 — Treino da rede base 80 → 64 → 32 → 1 (sem regularização)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-curvas-base.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
