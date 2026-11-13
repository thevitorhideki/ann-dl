"""Step 5: the ONLY step that touches the test set.

The configuration chosen on validation in step 4 is retrained with three seeds
(early stopping and threshold still decided on validation) and evaluated once on
the test set, next to the baselines fitted in step 1.

Run from the repository root (after step 4):
    python docs/projects/classification/code/step5_test.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from common import (FIGURES, RESULTS, Config, assess_success, choose_threshold,
                    count_parameters, evaluate, fit_pipeline, load_json, make_splits,
                    predict_proba, save_json, train_network)
from step1_baselines import build_baselines

SEEDS = [42, 43, 44]
METRICS = ["pr_auc", "roc_auc", "f1_yes", "precision_yes", "recall_yes", "f1_macro",
           "balanced_accuracy", "accuracy", "cost"]


def chosen_config() -> Config:
    best = load_json("step4_best.json")["best"]
    return Config(name="chosen", hidden=tuple(int(h) for h in best["hidden"].split("-")),
                  lr=best["lr"], dropout=best["dropout"], weight_decay=best["weight_decay"],
                  batchnorm=bool(best["batchnorm"]), pos_weight=best["pos_weight"])


def main() -> None:
    splits = make_splits()
    _, Xtr, Xva, Xte = fit_pipeline(splits)
    ytr, yva, yte = splits.y_train, splits.y_val, splits.y_test
    cfg = chosen_config()
    print("chosen configuration:", cfg.to_dict())

    # ---------------- network, three seeds ----------------
    runs, probas = [], {}
    for seed in SEEDS:
        model, hist = train_network(cfg, Xtr, ytr, Xva, yva, seed=seed)
        p_va, p_te = predict_proba(model, Xva), predict_proba(model, Xte)
        t = choose_threshold(yva, p_va)  # decided on validation, frozen for test
        m = {"seed": seed, "best_epoch": hist.best_epoch, "stopped_epoch": hist.stopped_epoch,
             "params": count_parameters(model),
             "train": evaluate(ytr, predict_proba(model, Xtr), t),
             "val": evaluate(yva, p_va, t), "test": evaluate(yte, p_te, t)}
        runs.append(m)
        probas[seed] = p_te
        if seed == SEEDS[0]:
            torch.save(model.state_dict(), RESULTS / "chosen_seed42.pt")
            np.save(RESULTS / "test_proba_seed42.npy", p_te)
            np.save(RESULTS / "val_proba_seed42.npy", p_va)
        print(f"seed {seed}: best epoch {hist.best_epoch} (stopped {hist.stopped_epoch}) | "
              f"t={t:.2f} | TEST PR-AUC {m['test']['pr_auc']:.4f} ROC-AUC "
              f"{m['test']['roc_auc']:.4f} F1(yes) {m['test']['f1_yes']:.4f}")

    test_df = pd.DataFrame([r["test"] for r in runs])
    summary = {k: {"mean": float(test_df[k].mean()), "std": float(test_df[k].std(ddof=1))}
               for k in METRICS}

    # ---------------- baselines, same test set ----------------
    base_results = {}
    for name, model in build_baselines().items():
        model.fit(Xtr, ytr)
        p_va, p_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
        t = 0.5 if name.startswith("Dummy") else choose_threshold(yva, p_va)
        base_results[name] = {"val": evaluate(yva, p_va, t), "test": evaluate(yte, p_te, t),
                              "proba_test": p_te}
        print(f"{name:26s} t={t:.2f} TEST PR-AUC {base_results[name]['test']['pr_auc']:.4f} "
              f"ROC-AUC {base_results[name]['test']['roc_auc']:.4f} F1(yes) "
              f"{base_results[name]['test']['f1_yes']:.4f}")

    # The best baseline is the one chosen on VALIDATION (step 1), not on test.
    candidates = {k: v for k, v in base_results.items() if not k.startswith("Dummy")}
    best_base = max(candidates, key=lambda k: candidates[k]["val"]["pr_auc"])
    base_pr = base_results[best_base]["test"]["pr_auc"]
    criterion = assess_success(summary["pr_auc"]["mean"], summary["pr_auc"]["std"], base_pr)
    print(f"\nnetwork TEST PR-AUC {summary['pr_auc']['mean']:.4f} ± {summary['pr_auc']['std']:.4f}")
    print(f"best baseline on validation: {best_base}; its TEST PR-AUC {base_pr:.4f}; "
          f"target {criterion['criterion_target']:.4f}; "
          f"relative gain {criterion['relative_gain']:+.2%}; "
          f"gap exceeds seed std: {criterion['gain_exceeds_seed_std']}; "
          f"criterion met: {criterion['criterion_met']}")

    save_json({
        "config": cfg.to_dict(), "seeds": SEEDS, "runs": runs, "summary": summary,
        "baselines": {k: {"val": v["val"], "test": v["test"]} for k, v in base_results.items()},
        "best_baseline": best_base, **criterion,
    }, "step5_test.json")

    # ---------------- Figure 3: PR curves on test ----------------
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for seed, color in zip(SEEDS, ["#D27632", "#E39A5E", "#F0BE8E"]):
        prec, rec, _ = precision_recall_curve(yte, probas[seed])
        ap = next(r["test"]["pr_auc"] for r in runs if r["seed"] == seed)
        ax.plot(rec, prec, color=color, label=f"Rede, seed {seed} (PR-AUC {ap:.3f})")
    for name, color in [("Gradient boosting (HGB)", "#5E9C5A"),
                        ("Logistic regression", "#466B91")]:
        prec, rec, _ = precision_recall_curve(yte, base_results[name]["proba_test"])
        ax.plot(rec, prec, color=color, ls="--",
                label=f"{name} (PR-AUC {base_results[name]['test']['pr_auc']:.3f})")
    ax.axhline(yte.mean(), color="gray", ls=":", label=f"Trivial = taxa de yes ({yte.mean():.3f})")
    ax.set(title="Figura 3 — Curva precisão × revocação no teste (classe yes)",
           xlabel="Revocação (recall)", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-pr-teste.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 4: confusion matrix, seed 42 ----------------
    r42 = runs[0]
    pred = (probas[SEEDS[0]] >= r42["test"]["threshold"]).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["no", "yes"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set(title=f"Figura 4 — Matriz de confusão no teste\n(seed 42, limiar "
                 f"{r42['test']['threshold']:.2f} escolhido na validação)",
           xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-confusao-teste.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
