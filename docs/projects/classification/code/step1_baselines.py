"""Step 1: baselines on the same pipeline, evaluated on VALIDATION only.

Run from the repository root:
    python docs/projects/classification/code/step1_baselines.py
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from common import SEED, choose_threshold, evaluate, fit_pipeline, make_splits, save_json


def build_baselines() -> dict:
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=5000, C=1.0),
        "Gradient boosting (HGB)": HistGradientBoostingClassifier(random_state=SEED),
    }


def main() -> None:
    splits = make_splits()
    print(f"train {len(splits.y_train):,} | val {len(splits.y_val):,} | test {len(splits.y_test):,}")
    print(f"positive rate train {splits.y_train.mean():.4f} | val {splits.y_val.mean():.4f}")
    _, Xtr, Xva, _ = fit_pipeline(splits)
    print(f"features after the EDA pipeline: {Xtr.shape[1]}")

    results = {
        "sizes": {"train": len(splits.y_train), "val": len(splits.y_val),
                  "test": len(splits.y_test), "n_features": int(Xtr.shape[1]),
                  "pos_rate_train": float(splits.y_train.mean())},
        "val": {},
    }
    for name, model in build_baselines().items():
        model.fit(Xtr, splits.y_train)
        p_val = model.predict_proba(Xva)[:, 1]
        # The dummy has constant probabilities: any threshold above them predicts "no".
        t = 0.5 if name.startswith("Dummy") else choose_threshold(splits.y_val, p_val)
        m = evaluate(splits.y_val, p_val, t)
        results["val"][name] = m
        print(f"{name:26s} PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
              f"t={t:.2f} F1(yes) {m['f1_yes']:.4f} | bal.acc {m['balanced_accuracy']:.4f} | "
              f"acc {m['accuracy']:.4f}")
    save_json(results, "step1_baselines.json")


if __name__ == "__main__":
    main()
