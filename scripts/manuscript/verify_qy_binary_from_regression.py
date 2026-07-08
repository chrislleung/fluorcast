from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

PRED_DIR = Path("outputs/paper_comparison/predictions")
OUT_DIR = Path("outputs/paper_comparison/qy_binary_from_regression")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.25

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None

rows = []

files = sorted(PRED_DIR.glob("quantum_yield__*__*__seed*.csv"))

if not files:
    raise SystemExit(f"No QY prediction files found in {PRED_DIR}")

for path in files:
    parts = path.stem.split("__")
    if len(parts) < 4:
        print(f"Skipping unexpected filename: {path}")
        continue

    target, model, split, seed = parts[0], parts[1], parts[2], parts[3]

    df = pd.read_csv(path)

    true_col = find_col(
        df,
        [
            "true_quantum_yield",
            "actual_quantum_yield",
            "quantum_yield_true",
            "y_true",
            "true",
            "actual",
        ],
    )

    pred_col = find_col(
        df,
        [
            "predicted_quantum_yield",
            "quantum_yield_pred",
            "prediction",
            "y_pred",
            "predicted",
            "pred",
        ],
    )

    if true_col is None or pred_col is None:
        print(f"Could not identify true/pred columns for: {path}")
        print("Columns:", list(df.columns))
        continue

    sub = df[[true_col, pred_col]].dropna().copy()

    y_true = (sub[true_col] > THRESHOLD).astype(int)
    y_pred = (sub[pred_col] > THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rows.append(
        {
            "model": model,
            "split": split,
            "seed": seed,
            "n": len(sub),
            "threshold": THRESHOLD,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_bright": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_bright": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "true_bright_fraction": y_true.mean(),
            "pred_bright_fraction": y_pred.mean(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "true_col": true_col,
            "pred_col": pred_col,
            "file": str(path),
        }
    )

results = pd.DataFrame(rows)

if results.empty:
    raise SystemExit("No QY binary classification metrics were calculated.")

per_seed_path = OUT_DIR / "qy_binary_from_regression_per_seed.csv"
summary_path = OUT_DIR / "qy_binary_from_regression_summary.csv"
best_path = OUT_DIR / "qy_binary_from_regression_best_by_split.csv"

results.to_csv(per_seed_path, index=False)

summary = (
    results
    .groupby(["split", "model"], as_index=False)
    .agg(
        n_mean=("n", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        precision_bright_mean=("precision_bright", "mean"),
        recall_bright_mean=("recall_bright", "mean"),
        true_bright_fraction_mean=("true_bright_fraction", "mean"),
        pred_bright_fraction_mean=("pred_bright_fraction", "mean"),
    )
)

summary = summary.sort_values(["split", "accuracy_mean"], ascending=[True, False])
summary.to_csv(summary_path, index=False)

best = summary.loc[summary.groupby("split")["accuracy_mean"].idxmax()].copy()
best = best.sort_values("split")
best.to_csv(best_path, index=False)

print("Saved:")
print(per_seed_path)
print(summary_path)
print(best_path)

print("\nBest model per split:")
print(
    best[
        [
            "split",
            "model",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "weighted_f1_mean",
            "n_mean",
        ]
    ].to_string(index=False)
)

print("\nExtraTrees rows:")
et = summary[summary["model"].str.lower().eq("extratrees")].sort_values("split")
print(
    et[
        [
            "split",
            "model",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "weighted_f1_mean",
            "n_mean",
        ]
    ].to_string(index=False)
)
