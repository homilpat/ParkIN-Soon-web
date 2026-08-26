"""Combined-source subject-grouped OOF training with a fixed 20% holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from train_spiral_grouped import (
    OUTPUT,
    SEED,
    fit_model,
    legacy_table,
    metrics,
    subject_predictions,
)
from validation_utils import threshold_at_minimum_sensitivity

TRACE_MANIFEST = Path(__file__).resolve().parent / "data/external/HandPD_trace_only/manifest.csv"
METADATA = OUTPUT / "drawing_trace_only_combined_80_20_metadata.json"


def trace_handpd_table():
    frame = pd.read_csv(TRACE_MANIFEST)
    trace_dir = TRACE_MANIFEST.parent
    frame["path"] = frame.path.map(lambda value: trace_dir / Path(value).name)
    if not frame.path.map(Path.exists).all():
        raise FileNotFoundError("Trace-only manifest and images do not match")
    return frame[["path", "label", "group"]]


def combined_table():
    handpd = trace_handpd_table().assign(source="HandPD_trace_only")
    legacy = legacy_table().assign(source="Legacy")
    handpd["group"] = "HandPD_" + handpd.group.astype(str)
    legacy["group"] = "Legacy_" + legacy.group.astype(str)
    frame = pd.concat([handpd, legacy], ignore_index=True)
    frame["stratum"] = frame.source + "_" + frame.label.astype(str)
    return frame


def subject_counts(frame):
    subjects = frame[["group", "source", "label"]].drop_duplicates()
    grouped = subjects.groupby(["source", "label"]).size()
    return {f"{source}_{label}": int(count) for (source, label), count in grouped.items()}


def split_development_holdout(frame):
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    development, holdout = next(splitter.split(frame, frame.stratum, frame.group))
    development, holdout = frame.iloc[development].copy(), frame.iloc[holdout].copy()
    if set(development.group) & set(holdout.group):
        raise RuntimeError("Subject leakage in 80:20 split")
    return development, holdout


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined = combined_table()
    development, holdout = split_development_holdout(combined)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1)
    oof_frames, models, fold_auc = [], [], []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(development, development.stratum, development.group), 1
    ):
        train, validation = development.iloc[train_idx], development.iloc[val_idx]
        if set(train.group) & set(validation.group):
            raise RuntimeError(f"Subject leakage in fold {fold}")
        print(f"\n===== COMBINED SUBJECT FOLD {fold}/5 =====")
        model = fit_model(train, validation)
        labels, probabilities = subject_predictions(model, validation)
        fold_auc.append(float(roc_auc_score(labels, probabilities)))
        for source in validation.source.unique():
            source_y, source_p = subject_predictions(model, validation[validation.source == source])
            oof_frames.append(pd.DataFrame(
                {"label": source_y, "probability": source_p, "source": source}
            ))
        models.append(model)
    oof = pd.concat(oof_frames, ignore_index=True)
    threshold = threshold_at_minimum_sensitivity(oof.label, oof.probability, 0.80)
    oof_result = metrics(oof.label.to_numpy(), oof.probability.to_numpy(), threshold)
    holdout_predictions = [subject_predictions(model, holdout) for model in models]
    holdout_y = holdout_predictions[0][0]
    holdout_p = np.mean([result[1] for result in holdout_predictions], axis=0)
    holdout_result = metrics(holdout_y, holdout_p, threshold)
    oof_by_source = {
        source: metrics(part.label.to_numpy(), part.probability.to_numpy(), threshold)
        for source, part in oof.groupby("source")
    }
    holdout_by_source = {}
    for source in holdout.source.unique():
        subset = holdout[holdout.source == source]
        predictions = [subject_predictions(model, subset) for model in models]
        holdout_by_source[source] = metrics(
            predictions[0][0], np.mean([item[1] for item in predictions], axis=0), threshold
        )
    metadata = {
        "model": "trace-only HandPD + legacy; frozen MobileNetV2, 32-unit head, 1x augmentation",
        "split": "source-class stratified subject-grouped 80:20",
        "all_subjects": int(combined.group.nunique()),
        "development": {"subjects": int(development.group.nunique()),
                        "images": len(development), "counts": subject_counts(development)},
        "holdout": {"subjects": int(holdout.group.nunique()),
                    "images": len(holdout), "counts": subject_counts(holdout)},
        "fold_subject_auc": fold_auc,
        "development_oof_threshold_setting": oof_result,
        "development_oof_by_source": oof_by_source,
        "internal_holdout": holdout_result,
        "internal_holdout_by_source": holdout_by_source,
        "deployment_status": "candidate_pending_external_validation_not_deployed",
    }
    for fold, model in enumerate(models, 1):
        model.save(OUTPUT / f"drawing_trace_only_combined_80_20_fold_{fold}.keras")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
