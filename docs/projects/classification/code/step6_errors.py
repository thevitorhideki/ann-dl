"""Step 6: error analysis.

Uses the test predictions saved in step 5 (seed 42, threshold chosen on validation)
to measure the error per subgroup and list the worst examples. Nothing is retrained
and no decision is taken from these numbers.

Permutation importance is computed on the VALIDATION set, at the level of the raw
columns (the column is shuffled BEFORE the pipeline, so all of its one-hot columns
move together).

Run from the repository root (after step 5):
    python docs/projects/classification/code/step6_errors.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from common import (FIGURES, RESULTS, SEED, MLP, fit_pipeline, load_json, make_splits,
                    predict_proba, save_json)

N_REPEATS = 5
RAW_FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing",
                "loan", "contact", "day", "month", "pdays", "previous", "poutcome"]


def subgroup_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(col, observed=True):
        pos = g[g.y == 1]
        flagged = g[g.pred == 1]
        rows.append({
            "feature": col, "group": str(value), "n": len(g), "positives": len(pos),
            "pos_rate": len(pos) / len(g),
            "error_rate": float((g.pred != g.y).mean()),
            "recall_yes": float((pos.pred == 1).mean()) if len(pos) else np.nan,
            "precision_yes": float((flagged.y == 1).mean()) if len(flagged) else np.nan,
            "pr_auc": float(average_precision_score(g.y, g.proba)) if 0 < len(pos) < len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    splits = make_splits()
    pre, Xtr, Xva, _ = fit_pipeline(splits)
    step5 = load_json("step5_test.json")
    run42 = step5["runs"][0]
    threshold = run42["test"]["threshold"]

    # ---------------- error per subgroup (test, seed 42) ----------------
    p_te = np.load(RESULTS / "test_proba_seed42.npy")
    df = splits.X_test.copy()
    df["y"], df["proba"] = splits.y_test, p_te
    df["pred"] = (p_te >= threshold).astype(int)
    df["age_band"] = pd.cut(df.age, [0, 29, 45, 60, 120],
                            labels=["≤29", "30–45", "46–60", "≥61"])
    tables = pd.concat([subgroup_table(df, c) for c in ["poutcome", "contact", "age_band", "month"]],
                       ignore_index=True)
    tables.to_csv(RESULTS / "step6_subgroups.csv", index=False)
    print(tables.round(3).to_string(index=False))
    worst = tables.loc[tables.error_rate.idxmax()]
    worst_error_subgroup = {
        "feature": worst["feature"], "group": worst["group"], "n": int(worst["n"]),
        "errors": int(round(worst["error_rate"] * worst["n"])),
        "error_rate": float(worst["error_rate"]),
    }
    print("\nHighest error rate:", worst_error_subgroup)

    # Worst examples: positives with the lowest score, negatives with the highest score.
    show = ["age", "job", "balance", "housing", "loan", "contact", "month", "poutcome",
            "previous", "pdays", "proba"]
    worst_fn = df[df.y == 1].nsmallest(5, "proba")[show]
    worst_fp = df[df.y == 0].nlargest(5, "proba")[show]
    worst_fn.to_csv(RESULTS / "step6_worst_fn.csv")
    worst_fp.to_csv(RESULTS / "step6_worst_fp.csv")
    print("\nWorst false negatives:\n", worst_fn.round(3).to_string())
    print("\nWorst false positives:\n", worst_fp.round(3).to_string())

    # ---------------- permutation importance (validation) ----------------
    cfg = step5["config"]
    model = MLP(Xtr.shape[1], tuple(cfg["hidden"]), cfg["dropout"], cfg["batchnorm"])
    model.load_state_dict(torch.load(RESULTS / "chosen_seed42.pt", weights_only=True))
    base_proba = predict_proba(model, Xva)
    base_ap = average_precision_score(splits.y_val, base_proba)
    rng = np.random.default_rng(SEED)
    imp = []
    for col in RAW_FEATURES:
        drops = []
        for _ in range(N_REPEATS):
            X_perm = splits.X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            p = predict_proba(model, pre.transform(X_perm).astype(np.float32))
            drops.append(base_ap - average_precision_score(splits.y_val, p))
        imp.append({"feature": col, "mean_drop": float(np.mean(drops)),
                    "std_drop": float(np.std(drops, ddof=1))})
    imp_df = pd.DataFrame(imp).sort_values("mean_drop", ascending=False)
    imp_df.to_csv(RESULTS / "step6_permutation.csv", index=False)
    print(f"\nvalidation PR-AUC (seed 42): {base_ap:.4f}")
    print(imp_df.round(4).to_string(index=False))

    save_json({"threshold": threshold, "val_pr_auc_seed42": float(base_ap),
               "test_error_rate": float((df.pred != df.y).mean()),
               "worst_error_subgroup": worst_error_subgroup}, "step6_errors.json")

    # ---------------- Figure 5: recall and precision per subgroup ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, col in zip(axes, ["poutcome", "contact", "age_band"]):
        t = tables[tables.feature == col]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.recall_yes, 0.4, color="#D27632", label="Revocação (yes)")
        ax.bar(x + 0.2, t.precision_yes, 0.4, color="#466B91", label="Precisão (yes)")
        ax.set_xticks(x, [f"{g}\n(n={n:,}, yes={p:.0%})" for g, n, p in
                          zip(t.group, t.n, t.pos_rate)], fontsize=8)
        ax.set(title=col, xlabel="Subgrupo (tamanho e taxa de yes no teste)", ylim=(0, 1))
    axes[0].set_ylabel("Valor no teste")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Figura 5 — Erro por subgrupo no teste (seed 42, limiar {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-subgrupos.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 6: permutation importance ----------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ordered = imp_df.iloc[::-1]
    ax.barh(ordered.feature, ordered.mean_drop, xerr=ordered.std_drop, color="#466B91",
            label=f"Queda média de PR-AUC ({N_REPEATS} permutações)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set(title="Figura 6 — Permutation importance na validação (seed 42)",
           xlabel="Queda de PR-AUC ao embaralhar a coluna bruta", ylabel="Feature bruta")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-permutation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
