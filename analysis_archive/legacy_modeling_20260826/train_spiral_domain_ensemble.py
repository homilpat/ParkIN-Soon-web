"""Train and save the locked five-fold domain-augmented screening ensemble."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from compare_spiral_domain_models import fit_predict, mobilenet_model
from train_spiral_combined_holdout import combined_table, split_development_holdout
from train_spiral_grouped import metrics
from validation_utils import threshold_at_minimum_sensitivity

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modeling"
METADATA = MODEL_DIR / "drawing_domain_ensemble_metadata.json"
SEED = 42


def source_metrics(scored, threshold):
    return {
        source: metrics(part.label.to_numpy(), part.probability.to_numpy(), threshold)
        for source, part in scored.groupby("source")
    }


def main():
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    development, _ = split_development_holdout(combined_table())
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1)
    oof, fold_auc, model_files = [], [], []
    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(development, development.stratum, development.group), 1
    ):
        train = development.iloc[train_idx]
        validation = development.iloc[validation_idx]
        if set(train.group) & set(validation.group):
            raise RuntimeError("Subject leakage")
        print(f"\n===== DEPLOYMENT CANDIDATE FOLD {fold}/5 =====")
        model, scored = fit_predict(mobilenet_model, train, validation, SEED + fold, 30)
        filename = f"drawing_domain_fold_{fold}.keras"
        model.save(MODEL_DIR / filename)
        model_files.append(filename)
        fold_auc.append(float(roc_auc_score(scored.label, scored.probability)))
        oof.append(scored)
        tf.keras.backend.clear_session()
    scored = pd.concat(oof, ignore_index=True)
    threshold = threshold_at_minimum_sensitivity(scored.label, scored.probability, .80)
    result = {
        "status": "active_screening_candidate_pending_independent_external_validation",
        "expression": "위험군 선별 보조 점수",
        "model_files": model_files,
        "preprocessing": "grayscale 3-channel; no guide alignment",
        "training_augmentation": "geometry, brightness, line width, blur, resolution, JPEG",
        "validation": "development-only subject-grouped 5-fold OOF; holdout not reused",
        "subjects": int(development.group.nunique()),
        "images": int(len(development)),
        "fold_auc": fold_auc,
        "fold_auc_std": float(np.std(fold_auc)),
        "overall": metrics(scored.label.to_numpy(), scored.probability.to_numpy(), threshold),
        "by_source": source_metrics(scored, threshold),
    }
    METADATA.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
