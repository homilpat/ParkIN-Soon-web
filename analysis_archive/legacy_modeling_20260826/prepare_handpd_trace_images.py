"""Create trace-only HandPD images by removing the black guide spiral."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from spiral_preprocessing import extract_blue_trace_rgb

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data/external/HandPD_Spiral"
METADATA = ROOT / "data/external/HandPD_Metadata/Spiral_HandPD.csv"
OUTPUT = ROOT / "data/external/HandPD_trace_only"
MANIFEST = OUTPUT / "manifest.csv"


def source_table():
    frame = pd.read_csv(METADATA).rename(columns={"_ID_EXAM": "ID_EXAM"})
    frame["label"] = frame.CLASS_TYPE.eq(2).astype(int)
    frame["group"] = frame.CLASS_TYPE.astype(str) + "_" + frame.ID_PATIENT.astype(str)
    frame["path"] = frame.apply(
        lambda row: IMAGES / ("SpiralPatients" if row.label else "SpiralControl") / row.IMAGE_NAME,
        axis=1,
    )
    return frame


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in source_table().iterrows():
        source = np.asarray(Image.open(row.path).convert("RGB"))
        # Color separation alone removes the black guide. Do not crop large
        # pathological deviations with a guide-centered corridor.
        trace, mask, raw_blue = extract_blue_trace_rgb(source, corridor_ratio=None)
        destination = OUTPUT / f"{index:04d}_{row.group}.png"
        Image.fromarray(trace).save(destination)
        raw_pixels = int((raw_blue > 0).sum())
        kept_pixels = int((mask > 0).sum())
        overlap = int(((mask > 0) & (raw_blue > 0)).sum())
        union = int(((mask > 0) | (raw_blue > 0)).sum())
        rows.append({"path": destination, "label": row.label, "group": row.group,
                     "source_path": row.path, "trace_pixels": kept_pixels,
                     "trace_fraction": float((mask > 0).mean()),
                     "raw_blue_pixels": raw_pixels,
                     "blue_recall": overlap / max(raw_pixels, 1),
                     "blue_precision": overlap / max(kept_pixels, 1),
                     "blue_iou": overlap / max(union, 1)})
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fractions = np.array([row["trace_fraction"] for row in rows])
    recall = np.array([row["blue_recall"] for row in rows])
    precision = np.array([row["blue_precision"] for row in rows])
    iou = np.array([row["blue_iou"] for row in rows])
    print(f"images={len(rows)} min={fractions.min():.6f} median={np.median(fractions):.6f} "
          f"max={fractions.max():.6f} low_count={(fractions < 0.002).sum()}")
    print(f"blue_recall min={recall.min():.6f} median={np.median(recall):.6f} mean={recall.mean():.6f}")
    print(f"blue_precision min={precision.min():.6f} median={np.median(precision):.6f} "
          f"mean={precision.mean():.6f}")
    print(f"blue_iou min={iou.min():.6f} median={np.median(iou):.6f} mean={iou.mean():.6f}")


if __name__ == "__main__":
    main()
