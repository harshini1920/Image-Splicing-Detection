"""
Classifier Training & Evaluation (pipeline step 10 / step 11)
---------------------------------------------------------------
Trains a gradient-boosted classifier on the feature vectors extracted by
src/training/extract_dataset_features.py.

Supports two evaluation protocols:

  --split stratified   Plain 5-fold stratified CV. Fast, but CASIA v2.0
                        reuses source photos across many tampered
                        examples, so near-duplicate images can land on
                        opposite sides of the split. This inflates AUC.

  --split grouped       (DEFAULT) Source-aware GroupKFold. CASIA filenames
                        embed source identifiers (e.g. an authentic
                        "Au_ani_00018.jpg" and a tampered
                        "Tp_D_CND_M_N_ani00018_sec00096_00138.jpg" both
                        reference "ani00018"). Images sharing an identifier
                        are grouped and kept in the SAME fold, so the model
                        is never evaluated on a near-duplicate of something
                        it trained on. This is the trustworthy result.

A gap between the two (stratified higher than grouped) is EXPECTED.

Usage:
    python train_classifier.py
    python train_classifier.py --split stratified
    python train_classifier.py --split both
"""

import argparse
import json
import os
import pickle
import re
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(_SRC_DIR, "..", "outputs", "evaluation", "casia_features.jsonl")
DEFAULT_MODEL_OUT = os.path.join(_SRC_DIR, "..", "models", "classifier.pkl")
DEFAULT_RESULTS_OUT = os.path.join(_SRC_DIR, "..", "outputs", "evaluation", "classifier_results.json")

N_FOLDS = 5
RANDOM_SEED = 42

_ID_TOKEN_RE = re.compile(r"^[a-zA-Z]{2,5}\d{4,6}$")
_LETTERS_ONLY_RE = re.compile(r"^[a-zA-Z]{2,5}$")
_DIGITS_ONLY_RE = re.compile(r"^\d{4,6}$")


def parse_source_identifiers(filename, is_authentic):
    stem = os.path.splitext(filename)[0]
    tokens = stem.split("_")
    ids = []

    if is_authentic:
        for i in range(len(tokens) - 1):
            if _LETTERS_ONLY_RE.match(tokens[i]) and _DIGITS_ONLY_RE.match(tokens[i + 1]):
                ids.append(tokens[i].lower() + tokens[i + 1])
    else:
        for t in tokens:
            if _ID_TOKEN_RE.match(t):
                ids.append(t.lower())

    return ids


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_source_groups(filenames, labels):
    uf = _UnionFind()
    per_image_ids = []

    for fname, label in zip(filenames, labels):
        ids = parse_source_identifiers(fname, is_authentic=(label == 0))
        per_image_ids.append(ids)
        for a, b in zip(ids, ids[1:]):
            uf.union(a, b)

    groups = []
    n_unparsed = 0
    for i, ids in enumerate(per_image_ids):
        if ids:
            groups.append(uf.find(ids[0]))
        else:
            n_unparsed += 1
            groups.append(f"__unparsed_{i}__")

    return groups, n_unparsed


def load_dataset(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    feature_names = sorted(records[0]["features"].keys())

    X = np.zeros((len(records), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(records), dtype=np.int64)
    filenames = []

    for i, rec in enumerate(records):
        feats = rec["features"]
        for j, name in enumerate(feature_names):
            X[i, j] = feats.get(name, 0.0)
        y[i] = rec["label"]
        filenames.append(rec["filename"])

    return X, y, feature_names, filenames


def _make_classifier(n_train, seed=RANDOM_SEED):
    use_early_stopping = n_train >= 200
    return HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=use_early_stopping,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=seed,
    )


def train_and_evaluate(X, y, split_type, groups=None, n_folds=N_FOLDS, seed=RANDOM_SEED):
    if split_type == "grouped":
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    oof_proba = np.zeros(len(y), dtype=np.float64)
    fold_aucs = []
    fold_thresholds = []

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    weight_pos = len(y) / (2 * n_pos)
    weight_neg = len(y) / (2 * n_neg)
    sample_weight_full = np.where(y == 1, weight_pos, weight_neg)

    for fold_i, (train_idx, val_idx) in enumerate(split_iter, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weight_full[train_idx]

        if len(np.unique(y_val)) < 2:
            print(f"  Fold {fold_i}/{n_folds}: SKIPPED scoring (only one class in validation fold)")
            clf = _make_classifier(len(X_train), seed)
            clf.fit(X_train, y_train, sample_weight=w_train)
            oof_proba[val_idx] = clf.predict_proba(X_val)[:, 1]
            continue

        clf = _make_classifier(len(X_train), seed)
        clf.fit(X_train, y_train, sample_weight=w_train)

        val_proba = clf.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_aucs.append(fold_auc)

        thresholds = np.linspace(0.05, 0.95, 37)
        best_f1, best_t = -1, 0.5
        for t in thresholds:
            f1 = f1_score(y_val, (val_proba >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        fold_thresholds.append(best_t)

        print(f"  Fold {fold_i}/{n_folds}: AUC={fold_auc:.4f}  "
              f"best_threshold={best_t:.2f}  (train={len(train_idx)}, val={len(val_idx)})")

    median_threshold = float(np.median(fold_thresholds)) if fold_thresholds else 0.5
    overall_auc = roc_auc_score(y, oof_proba)
    oof_pred = (oof_proba >= median_threshold).astype(int)

    metrics = {
        "split_type": split_type,
        "auc": float(overall_auc),
        "f1": float(f1_score(y, oof_pred)),
        "accuracy": float(accuracy_score(y, oof_pred)),
        "precision": float(precision_score(y, oof_pred)),
        "recall": float(recall_score(y, oof_pred)),
        "median_threshold": median_threshold,
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
    }
    return oof_proba, oof_pred, metrics, sample_weight_full


def _print_metrics(metrics, title):
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"  AUC:        {metrics['auc']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Threshold:  {metrics['median_threshold']:.2f}")
    if metrics["fold_aucs"]:
        print(f"  Per-fold AUC: {[round(a,4) for a in metrics['fold_aucs']]}")
        print(f"  Fold AUC mean +/- std: {metrics['fold_auc_mean']:.4f} +/- {metrics['fold_auc_std']:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the splicing classifier")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT)
    parser.add_argument("--results-out", default=DEFAULT_RESULTS_OUT)
    parser.add_argument("--split", choices=["grouped", "stratified", "both"], default="grouped")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input} ...")
    t0 = time.time()
    X, y, feature_names, filenames = load_dataset(args.input)
    print(f"Loaded {len(y)} images x {len(feature_names)} features in {time.time()-t0:.1f}s")
    print(f"Class balance: {int((y==0).sum())} authentic, {int((y==1).sum())} tampered\n")

    groups, n_unparsed = build_source_groups(filenames, y)
    n_unique_groups = len(set(groups))
    print(f"Source-aware grouping: {n_unique_groups} unique source groups "
          f"({n_unparsed} images had unparseable filenames -> isolated groups)\n")

    all_metrics = {}

    if args.split in ("grouped", "both"):
        print("Running 5-fold GROUP-aware cross-validation (source-aware, paper-matching) ...")
        t0 = time.time()
        _, _, metrics_grouped, sample_weight = train_and_evaluate(X, y, "grouped", groups=groups)
        print(f"Finished in {time.time()-t0:.1f}s\n")
        _print_metrics(metrics_grouped, "RESULTS -- GROUPED (source-aware, trustworthy)")
        all_metrics["grouped"] = metrics_grouped

    if args.split in ("stratified", "both"):
        print("\nRunning 5-fold STRATIFIED cross-validation (plain, optimistic comparison) ...")
        t0 = time.time()
        _, _, metrics_strat, sample_weight = train_and_evaluate(X, y, "stratified")
        print(f"Finished in {time.time()-t0:.1f}s\n")
        _print_metrics(metrics_strat, "RESULTS -- STRATIFIED (plain, likely optimistic)")
        all_metrics["stratified"] = metrics_strat

    if args.split == "both":
        gap = all_metrics["stratified"]["auc"] - all_metrics["grouped"]["auc"]
        print(f"\nGap (stratified - grouped) AUC: {gap:+.4f}  "
              f"(a positive gap here is expected -- it's the leakage the grouped split removes)")

    print("\nTraining final model on full dataset for deployment...")
    final_clf = _make_classifier(len(X))
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    sample_weight_full = np.where(
        y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg)
    )
    final_clf.fit(X, y, sample_weight=sample_weight_full)

    primary = all_metrics.get("grouped", all_metrics.get("stratified"))
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": final_clf,
            "feature_names": feature_names,
            "threshold": primary["median_threshold"],
        }, f)
    print(f"Final model saved to {args.model_out}")

    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {args.results_out}")


if __name__ == "__main__":
    main()