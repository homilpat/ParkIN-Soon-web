"""Original spiral CNN with subject-safe validation and external testing."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from validation_utils import threshold_at_minimum_sensitivity

ROOT = Path(__file__).resolve().parent
HANDPD = ROOT / "data/external/HandPD_Spiral"
HANDPD_META = ROOT / "data/external/HandPD_Metadata/Spiral_HandPD.csv"
LEGACY = ROOT / "data/parkinsons-drawings/parkinsons-drawings/spiral"
OUTPUT = ROOT / "modeling/candidates"
SIZE, BATCH, SEED, FOLDS = (224, 224), 8, 42, 5
AUGMENTATION_EXPOSURE = 1


def grayscale_3ch(image):
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return np.repeat(gray[..., None], 3, axis=-1)


TRAIN_AUG = ImageDataGenerator(
    rescale=1.0 / 255, preprocessing_function=grayscale_3ch,
    rotation_range=10, width_shift_range=0.08, height_shift_range=0.08,
    zoom_range=0.08, shear_range=0.08, brightness_range=(0.85, 1.15),
)
EVAL_AUG = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=grayscale_3ch)


def handpd_table():
    frame = pd.read_csv(HANDPD_META).rename(columns={"_ID_EXAM": "ID_EXAM"})
    frame["label"] = frame.CLASS_TYPE.eq(2).astype(int)
    frame["group"] = frame.CLASS_TYPE.astype(str) + "_" + frame.ID_PATIENT.astype(str)
    frame["path"] = frame.apply(
        lambda row: HANDPD / ("SpiralPatients" if row.label else "SpiralControl") / row.IMAGE_NAME,
        axis=1,
    )
    if not frame.path.map(Path.exists).all():
        raise FileNotFoundError("HandPD metadata and images do not match")
    return frame[["path", "label", "group"]]


def legacy_table():
    rows = []
    for split in ("training", "testing"):
        for folder, label in (("healthy", 0), ("parkinson", 1)):
            for path in (LEGACY / split / folder).glob("*.png"):
                match = re.match(r"^(V\d+[HP])", path.stem, re.IGNORECASE)
                if match:
                    rows.append((path, label, match.group(1).upper()))
    return pd.DataFrame(rows, columns=["path", "label", "group"])


def generator(frame, training):
    data = frame.assign(path=frame.path.astype(str), label=frame.label.astype(str))
    if training:
        data = pd.concat([data] * AUGMENTATION_EXPOSURE, ignore_index=True)
    return (TRAIN_AUG if training else EVAL_AUG).flow_from_dataframe(
        data, x_col="path", y_col="label", target_size=SIZE, batch_size=BATCH,
        class_mode="binary", shuffle=training, seed=SEED if training else None,
    )


def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-2))(x)
    x = Dropout(0.3)(x)
    return tf.keras.Model(base.input, Dense(1, activation="sigmoid")(x))


def callbacks(patience):
    return [
        EarlyStopping(monitor="val_auc", patience=patience, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7),
    ]


def fit_model(train, validation):
    train_gen, validation_gen = generator(train, True), generator(validation, False)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train.label)
    model = build_model()
    model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(train_gen, validation_data=validation_gen, epochs=30, callbacks=callbacks(6),
              class_weight=dict(enumerate(weights)), verbose=2)
    return model


def subject_predictions(model, frame):
    probabilities = model.predict(generator(frame, False), verbose=0).ravel()
    scored = frame.assign(probability=probabilities)
    grouped = scored.groupby("group", as_index=False).agg(
        probability=("probability", "mean"), label=("label", "first")
    )
    return grouped.label.to_numpy(), grouped.probability.to_numpy()


def metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(labels, probabilities)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "f1": float(f1_score(labels, predicted)), "threshold": float(threshold),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_data, external = handpd_table(), legacy_table()
    labels, groups = main_data.label.to_numpy(), main_data.group.to_numpy()
    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    subject_oof, models, fold_auc_scores = [], [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(main_data, labels, groups), 1):
        train, validation = main_data.iloc[train_idx], main_data.iloc[val_idx]
        assert set(train.group).isdisjoint(validation.group)
        print(f"\n===== SUBJECT FOLD {fold}/{FOLDS} =====")
        model = fit_model(train, validation)
        fold_y, fold_p = subject_predictions(model, validation)
        fold_auc = roc_auc_score(fold_y, fold_p)
        subject_oof.append(pd.DataFrame({"label": fold_y, "probability": fold_p}))
        models.append(model)
        fold_auc_scores.append(float(fold_auc))
    oof = pd.concat(subject_oof, ignore_index=True)
    threshold = threshold_at_minimum_sensitivity(oof.label, oof.probability, 0.80)
    internal = metrics(oof.label.to_numpy(), oof.probability.to_numpy(), threshold)
    external_predictions = [subject_predictions(model, external) for model in models]
    external_y = external_predictions[0][0]
    external_p = np.mean([prediction[1] for prediction in external_predictions], axis=0)
    external_result = metrics(external_y, external_p, threshold)
    metadata = {
        "model": "frozen MobileNetV2, constrained 32-unit head, 3x augmentation, fold ensemble",
        "validation_unit": "subject", "training": {"source": "HandPD", "subjects": 54,
        "images": len(main_data), "augmentation_exposure_per_epoch": AUGMENTATION_EXPOSURE},
        "fold_subject_auc": fold_auc_scores,
        "internal_grouped_oof_threshold_setting": internal,
        "external_legacy_28_subjects": external_result,
        "deployment_status": "candidate_pending_review_not_deployed",
        "deployment_note": (
            f"External sensitivity={external_result['sensitivity']:.4f}, "
            f"specificity={external_result['specificity']:.4f}; manual review required."
        ),
    }
    for fold, model in enumerate(models, 1):
        model.save(OUTPUT / f"drawing_constrained_aug3_fold_{fold}.keras")
    (OUTPUT / "drawing_constrained_aug3_ensemble_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
