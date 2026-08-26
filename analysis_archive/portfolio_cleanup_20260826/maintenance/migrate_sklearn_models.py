"""Reserialize deployed sklearn models and verify prediction preservation."""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modeling"
REPORT_PATH = MODEL_DIR / "sklearn_model_migration_1_9.json"
MODEL_SPECS = {
    "olfactory_bowel": {
        "path": MODEL_DIR / "olf_con_model.pkl",
        "source_version": "1.5.0",
        "features": 15,
    },
    "kinematic": {
        "path": MODEL_DIR / "drawing_kinematic_model.pkl",
        "source_version": "1.7.2",
        "features": 12,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def estimator_from_artifact(artifact: object) -> object:
    if not isinstance(artifact, dict):
        return artifact
    for key in ("pipeline", "model", "estimator"):
        if key in artifact:
            return artifact[key]
    raise KeyError("Model dictionary has no pipeline/model/estimator entry")


def prediction(estimator: object, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        values = estimator.predict_proba(matrix)
        return np.asarray(values, dtype=float)
    return np.asarray(estimator.predict(matrix), dtype=float)


def synthetic_matrix(feature_count: int) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, feature_count)
    return np.vstack(
        [
            np.zeros(feature_count),
            np.ones(feature_count),
            grid,
            grid[::-1],
            np.arange(feature_count) % 2,
            (np.arange(feature_count) % 3) / 2,
        ]
    ).astype(float)


def migrate_one(name: str, spec: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    temporary = path.with_suffix(path.suffix + ".migrating")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(path)
    source_warnings = [str(item.message) for item in caught]
    matrix = synthetic_matrix(int(spec["features"]))
    before = prediction(estimator_from_artifact(artifact), matrix)
    original_hash = sha256(path)

    try:
        joblib.dump(artifact, temporary)
        with warnings.catch_warnings(record=True) as caught_reload:
            warnings.simplefilter("always", InconsistentVersionWarning)
            reloaded = joblib.load(temporary)
        version_warnings = [str(item.message) for item in caught_reload]
        if version_warnings:
            raise RuntimeError(f"Version warning remained after migration: {version_warnings}")
        after = prediction(estimator_from_artifact(reloaded), matrix)
        np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-12)
        max_diff = float(np.max(np.abs(before - after)))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "model": name,
        "source_sklearn_version": spec["source_version"],
        "original_sha256": original_hash,
        "migrated_sha256": sha256(path),
        "equivalence_samples": int(matrix.shape[0]),
        "max_abs_prediction_diff": max_diff,
        "source_version_warning_count": len(source_warnings),
        "reload_version_warning_count": 0,
    }


def main() -> None:
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(f"Run with scikit-learn 1.9.0, got {sklearn.__version__}")
    results = [migrate_one(name, spec) for name, spec in MODEL_SPECS.items()]
    report = {
        "migrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "scope": "serialization compatibility migration; model fitting and metrics not rerun",
        "models": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
