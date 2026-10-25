"""Train, compare and export a breast cancer triage model.

Label convention (kept exactly as in the raw data, never remapped):
    target 0 = malignant  <- POSITIVE class for every "malignant" metric
    target 1 = benign
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "target"
MALIGNANT_LABEL = 0  # positive class
RECALL_GATE = 0.95  # minimum test-set malignant recall for the selected model
GRID_KEY = "grid"  # params.yaml key holding the search space (never passed to the classifier)
STEP = "clf"  # pipeline step name of the classifier, used to prefix grid keys (clf__C)

# Candidate name (= params.yaml block) -> classifier class
CLASSIFIERS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the breast cancer triage model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/breast_cancer.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", type=str, default="1.0.0")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    return parser.parse_args()


def load_params(path: Path) -> dict[str, Any]:
    """Load the training parameters from a YAML file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and split it into features (original column order) and target."""
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' not found in {path}")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def split_model_params(block: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list]]:
    """Separate fixed classifier arguments from the search grid."""
    fixed = {k: v for k, v in block.items() if k != GRID_KEY}
    grid = {k: v if isinstance(v, list) else [v] for k, v in block.get(GRID_KEY, {}).items()}
    return fixed, grid


def build_candidates(
    params: dict[str, Any], seed: int
) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Build (scaler + classifier Pipeline, prefixed param grid) for every candidate."""
    candidates: dict[str, tuple[Pipeline, dict[str, list]]] = {}
    for name, classifier_cls in CLASSIFIERS.items():
        fixed, grid = split_model_params(params[name])
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (STEP, classifier_cls(random_state=seed, **fixed)),
            ]
        )
        # Prefix grid keys with the pipeline step name, e.g. C -> clf__C
        param_grid = {f"{STEP}__{k}": v for k, v in grid.items()}
        candidates[name] = (pipeline, param_grid)
    return candidates


def make_cv(folds: int, seed: int) -> StratifiedKFold:
    """Shared CV splitter: same folds for tuning and for reported CV metrics."""
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def recall_scorer() -> Any:
    """Recall with malignant (target 0) as the positive class."""
    return make_scorer(recall_score, pos_label=MALIGNANT_LABEL)


def tuning_scoring() -> dict[str, Any]:
    """Scorers used by GridSearchCV."""
    return {"recall_malignant": recall_scorer(), "roc_auc": "roc_auc"}


def cv_scoring() -> dict[str, Any]:
    """Scorers for cross-validation; recall treats malignant (0) as positive."""
    return {
        "recall_malignant": recall_scorer(),
        "accuracy": "accuracy",
        # ROC AUC is symmetric: AUC(P(benign), benign+) == AUC(P(malignant), malignant+)
        "roc_auc": "roc_auc",
    }


def refit_by_recall_then_auc(cv_results: dict[str, Any]) -> int:
    """GridSearchCV refit callable: highest mean recall, ties broken by mean ROC AUC."""
    recall = cv_results["mean_test_recall_malignant"]
    auc = cv_results["mean_test_roc_auc"]
    # max() keeps the first index on a full tie, so the result is deterministic
    return max(range(len(recall)), key=lambda i: (recall[i], auc[i]))


def tune_model(
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> GridSearchCV:
    """Grid-search one candidate on the training data and refit the best pipeline."""
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=tuning_scoring(),
        refit=refit_by_recall_then_auc,
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def strip_prefix(best_params: dict[str, Any]) -> dict[str, Any]:
    """Turn {'clf__C': 1.0} into {'C': 1.0} to match params.yaml naming."""
    return {k.split("__", 1)[1]: v for k, v in best_params.items()}


def cross_validate_model(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> dict[str, float]:
    """Run stratified k-fold CV on the given (training) data; return mean/std per metric."""
    results = cross_validate(model, X, y, cv=cv, scoring=cv_scoring())

    summary: dict[str, float] = {}
    for metric in cv_scoring():
        scores = results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(scores))
        summary[f"{metric}_std"] = float(np.std(scores))
    return summary


def evaluate_on_test(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute held-out test metrics (reporting only, never used for selection)."""
    y_pred = model.predict(X)
    # Probability of class 1; AUC is identical when using P(class 0) with 0 as positive
    benign_idx = list(model.classes_).index(1)
    y_proba = model.predict_proba(X)[:, benign_idx]

    return {
        "recall_malignant": float(recall_score(y, y_pred, pos_label=MALIGNANT_LABEL)),
        "precision_malignant": float(
            precision_score(y, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


def select_best(cv_results: dict[str, dict[str, float]]) -> str:
    """Pick the highest mean CV malignant recall; break ties with mean CV ROC AUC."""
    return max(
        cv_results,
        key=lambda name: (
            cv_results[name]["recall_malignant_mean"],
            cv_results[name]["roc_auc_mean"],
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_comparison(
    cv_results: dict[str, dict[str, float]],
    test_results: dict[str, dict[str, float]],
    best_params: dict[str, dict[str, Any]],
    selected: str,
) -> None:
    """Print one comparison table of CV and test metrics for every candidate."""
    header = (
        f"{'model':<22}{'cv_recall':>10}{'cv_acc':>10}{'cv_auc':>10}"
        f"{'test_recall':>13}{'test_acc':>10}{'test_auc':>10}  selected"
    )
    print("\nModel comparison (recall = malignant recall, positive class = target 0)")
    print(header)
    print("-" * len(header))
    for name in cv_results:
        cv, test = cv_results[name], test_results[name]
        mark = "  <==" if name == selected else ""
        print(
            f"{name:<22}"
            f"{cv['recall_malignant_mean']:>10.4f}"
            f"{cv['accuracy_mean']:>10.4f}"
            f"{cv['roc_auc_mean']:>10.4f}"
            f"{test['recall_malignant']:>13.4f}"
            f"{test['accuracy']:>10.4f}"
            f"{test['roc_auc']:>10.4f}"
            f"{mark}"
        )
    print("\nBest parameters (GridSearchCV on training split):")
    for name, best in best_params.items():
        print(f"  {name:<20} {best}")
    print(f"\nSelected model (by CV only): {selected}\n")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a pretty-printed JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_model_card(
    model: Pipeline,
    feature_names: list[str],
    model_version: str,
    test_metrics: dict[str, float],
    cv_metrics: dict[str, float],
    best_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Collect metadata describing the selected model."""
    return {
        "model_version": model_version,
        "model_family": type(model.named_steps[STEP]).__name__,
        "best_params": best_params,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "positive_class": "malignant (target 0)",
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    }


def build_metrics(
    selected: str,
    selected_best_params: dict[str, Any],
    cv_results: dict[str, dict[str, float]],
    test_results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Flat-ish metrics dict (one level per model) suitable for `dvc metrics`."""
    metrics: dict[str, Any] = {
        "selected_model": selected,
        "selected_best_params": selected_best_params,
    }
    for name in cv_results:
        entry = {f"cv_{k}": v for k, v in cv_results[name].items()}
        entry.update({f"test_{k}": v for k, v in test_results[name].items()})
        metrics[name] = entry
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    seed: int = params["seed"]

    # Load data and make a stratified hold-out split
    X, y = load_data(args.data_path)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=seed
    )

    cv = make_cv(params["cv_folds"], seed)

    # Tune every candidate on the training split only, then report CV metrics
    # for its best parameters on the same folds
    fitted: dict[str, Pipeline] = {}
    best_params: dict[str, dict[str, Any]] = {}
    cv_results: dict[str, dict[str, float]] = {}
    for name, (pipeline, param_grid) in build_candidates(params, seed).items():
        print(f"Tuning {name} over {param_grid} ...")
        search = tune_model(pipeline, param_grid, X_train, y_train, cv)
        fitted[name] = search.best_estimator_  # already refit on the full training split
        best_params[name] = strip_prefix(search.best_params_)
        cv_results[name] = cross_validate_model(search.best_estimator_, X_train, y_train, cv)

    selected = select_best(cv_results)

    # Evaluate every tuned candidate on the test set (reporting only)
    test_results = {
        name: evaluate_on_test(model, X_test, y_test) for name, model in fitted.items()
    }

    print_comparison(cv_results, test_results, best_params, selected)

    # Quality gate: stop before writing any artifact
    selected_recall = test_results[selected]["recall_malignant"]
    if selected_recall < RECALL_GATE:
        print(
            f"ERROR: quality gate failed - '{selected}' test malignant recall "
            f"{selected_recall:.4f} < {RECALL_GATE}. No artifacts written.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write artifacts
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(fitted[selected], output_dir / "model.joblib")
    write_json(
        output_dir / "model_card.json",
        build_model_card(
            fitted[selected],
            feature_names,
            args.model_version,
            test_results[selected],
            cv_results[selected],
            best_params[selected],
            seed,
        ),
    )
    write_json(
        output_dir / "metrics.json",
        build_metrics(selected, best_params[selected], cv_results, test_results),
    )

    print(f"Quality gate passed (test malignant recall = {selected_recall:.4f}).")
    print(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
