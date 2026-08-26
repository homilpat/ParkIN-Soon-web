"""Repeated subject-grouped comparison of domain augmentation and a small CNN."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train_spiral_combined_holdout import combined_table, split_development_holdout
from train_spiral_grouped import BATCH, SIZE, metrics
from validation_utils import threshold_at_minimum_sensitivity

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling/candidates/drawing_domain_model_comparison.json"
SEEDS = (42, 142, 242)


def grayscale_3ch(image):
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


def domain_preprocess(image):
    """Simulate line-width, resolution, blur, and compression differences."""
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    if np.random.random() < 0.5:
        operation = cv2.erode if np.random.random() < 0.5 else cv2.dilate
        gray = operation(gray, np.ones((2, 2), np.uint8), iterations=1)
    if np.random.random() < 0.35:
        gray = cv2.GaussianBlur(gray, (3, 3), np.random.uniform(0.2, 0.8))
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.55, 0.9)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    if np.random.random() < 0.4:
        quality = int(np.random.randint(55, 91))
        ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


EVAL_GEN = ImageDataGenerator(rescale=1 / 255, preprocessing_function=grayscale_3ch)


def train_datagen():
    return ImageDataGenerator(
        rescale=1 / 255, preprocessing_function=domain_preprocess,
        rotation_range=10, width_shift_range=.08, height_shift_range=.08,
        zoom_range=.08, shear_range=.08, brightness_range=(.85, 1.15),
    )


def generator(frame, training, seed):
    data = frame.assign(path=frame.path.astype(str), label=frame.label.astype(str))
    factory = train_datagen() if training else EVAL_GEN
    return factory.flow_from_dataframe(
        data, x_col="path", y_col="label", target_size=SIZE, batch_size=BATCH,
        class_mode="binary", shuffle=training, seed=seed if training else None,
    )


def mobilenet_model():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*SIZE, 3)
    )
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(.5)(x)
    x = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-2))(x)
    x = Dropout(.3)(x)
    return tf.keras.Model(base.input, Dense(1, activation="sigmoid")(x))


def small_cnn_model():
    inputs = tf.keras.Input(shape=(*SIZE, 3))
    x = inputs
    for filters in (16, 32, 64, 96):
        x = Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(.4)(x)
    x = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    return tf.keras.Model(inputs, Dense(1, activation="sigmoid")(x))


def fit_predict(builder, train, validation, seed, epochs):
    model = builder()
    model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train.label)
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=6, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=.5, min_lr=1e-7),
    ]
    model.fit(generator(train, True, seed), validation_data=generator(validation, False, seed),
              epochs=epochs, class_weight=dict(enumerate(weights)), callbacks=callbacks, verbose=2)
    probabilities = model.predict(generator(validation, False, seed), verbose=0).ravel()
    scored = validation.assign(probability=probabilities)
    subject_scores = scored.groupby(["group", "source"], as_index=False).agg(
        probability=("probability", "mean"), label=("label", "first")
    )
    return model, subject_scores


def evaluate_seed(name, builder, development, seed, epochs):
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed + 1)
    folds, oof = [], []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(development, development.stratum, development.group), 1
    ):
        train, validation = development.iloc[train_idx], development.iloc[val_idx]
        if set(train.group) & set(validation.group):
            raise RuntimeError("Subject leakage")
        print(f"\n===== {name} seed={seed} fold={fold}/5 =====")
        model, scored = fit_predict(builder, train, validation, seed + fold, epochs)
        folds.append(float(roc_auc_score(scored.label, scored.probability)))
        oof.append(scored)
        tf.keras.backend.clear_session()
    scored = pd.concat(oof, ignore_index=True)
    threshold = threshold_at_minimum_sensitivity(scored.label, scored.probability, .80)
    overall = metrics(scored.label.to_numpy(), scored.probability.to_numpy(), threshold)
    by_source = {
        source: metrics(part.label.to_numpy(), part.probability.to_numpy(), threshold)
        for source, part in scored.groupby("source")
    }
    return {"seed": seed, "fold_auc": folds, "fold_auc_std": float(np.std(folds)),
            "overall": overall, "by_source": by_source,
            "worst_source_auc": min(item["auc"] for item in by_source.values())}


def aggregate(runs):
    fields = ("auc", "sensitivity", "specificity")
    summary = {}
    for field in fields:
        values = [run["overall"][field] for run in runs]
        summary[field] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    worst = [run["worst_source_auc"] for run in runs]
    summary["worst_source_auc"] = {"mean": float(np.mean(worst)), "std": float(np.std(worst))}
    summary["fold_auc_std_mean"] = float(np.mean([run["fold_auc_std"] for run in runs]))
    return summary


def main():
    combined = combined_table()
    development, _ = split_development_holdout(combined)
    experiments = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    for name, builder, epochs in (("domain_aug_mobilenet", mobilenet_model, 30),
                                  ("domain_aug_small_cnn", small_cnn_model, 20)):
        if name in experiments:
            print(f"Skipping completed experiment: {name}")
            continue
        runs = []
        for seed in SEEDS:
            tf.keras.utils.set_random_seed(seed)
            np.random.seed(seed)
            runs.append(evaluate_seed(name, builder, development, seed, epochs))
        experiments[name] = {"runs": runs, "summary": aggregate(runs)}
        OUTPUT.write_text(json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(experiments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
