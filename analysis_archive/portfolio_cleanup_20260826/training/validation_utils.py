import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def youden_threshold(y_true, probabilities) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])


def threshold_at_minimum_sensitivity(
    y_true, probabilities, minimum_sensitivity: float = 0.80
) -> float:
    """Choose maximum specificity subject to sensitivity meeting the target."""
    if not 0.0 < minimum_sensitivity <= 1.0:
        raise ValueError("minimum_sensitivity must be in (0, 1]")
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    candidates = np.isfinite(thresholds) & (tpr >= minimum_sensitivity)
    if not candidates.any():
        # ROC always contains an all-positive operating point, but fail safely if
        # an unusual estimator produces no finite candidate.
        return 0.0
    candidate_indices = np.flatnonzero(candidates)
    best_specificity = np.max(1.0 - fpr[candidate_indices])
    best_indices = candidate_indices[np.isclose(1.0 - fpr[candidate_indices], best_specificity)]
    # If specificity ties, retain the candidate with the greatest sensitivity.
    best_index = best_indices[np.argmax(tpr[best_indices])]
    return float(thresholds[best_index])


def nested_threshold_evaluation(
    estimator,
    X,
    y,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
    minimum_sensitivity: float | None = None,
) -> Dict:
    """임계값 선택과 평가 데이터를 분리한 nested OOF 검증입니다."""
    X = np.asarray(X)
    y = np.asarray(y)
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=random_state)
    oof_probabilities = np.full(len(y), np.nan, dtype=float)
    oof_predictions = np.full(len(y), -1, dtype=int)
    fold_thresholds = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=random_state + fold)
        inner_probabilities = cross_val_predict(
            clone(estimator),
            X[train_idx],
            y[train_idx],
            cv=inner,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        if minimum_sensitivity is None:
            threshold = youden_threshold(y[train_idx], inner_probabilities)
        else:
            threshold = threshold_at_minimum_sensitivity(
                y[train_idx], inner_probabilities, minimum_sensitivity
            )
        fitted = clone(estimator).fit(X[train_idx], y[train_idx])
        probabilities = fitted.predict_proba(X[test_idx])[:, 1]
        oof_probabilities[test_idx] = probabilities
        oof_predictions[test_idx] = (probabilities >= threshold).astype(int)
        fold_thresholds.append(threshold)

    tn, fp, fn, tp = confusion_matrix(y, oof_predictions, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, oof_probabilities)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "deployment_threshold": float(np.median(fold_thresholds)),
        "fold_thresholds": [float(value) for value in fold_thresholds],
        "threshold_rule": (
            "youden"
            if minimum_sensitivity is None
            else f"maximum specificity with inner-CV sensitivity >= {minimum_sensitivity:.2f}"
        ),
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
    }


def update_fusion_config(config_path, modality: str, auc: float, threshold: float) -> None:
    """재학습 결과를 서비스 fusion 설정에 원자적으로 반영합니다."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if modality not in {"olf", "img", "kin"}:
        raise ValueError("알 수 없는 modality입니다.")
    config.setdefault("auc_scores", {})[modality] = float(auc)
    config.setdefault("thresholds", {})[modality] = float(threshold)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)
