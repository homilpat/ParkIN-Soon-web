"""Render UCI 395 static-spiral tablet trajectories as classifier images."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
UCI = ROOT / "data/external/UCI_395"
OUTPUT = UCI / "rendered_static_spiral"
MANIFEST = OUTPUT / "manifest.csv"
CANVAS = 224
TABLET_MIN, TABLET_MAX = 0.0, 500.0


def sources():
    yield from ((path, 0, "hw_control") for path in (UCI / "hw_dataset/control").glob("*.txt"))
    yield from ((path, 1, "hw_parkinson") for path in (UCI / "hw_dataset/parkinson").glob("*.txt"))
    yield from ((path, 1, "new_parkinson") for path in (UCI / "new_dataset/parkinson").glob("*.txt"))


def read_static_points(path):
    points = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split(";")
            if len(fields) != 7 or int(fields[6]) != 0:
                continue
            x, y, pressure = float(fields[0]), float(fields[1]), float(fields[3])
            points.append((x, y, pressure > 0))
    return points


def pixel(value):
    clipped = min(TABLET_MAX, max(TABLET_MIN, value))
    return round((clipped - TABLET_MIN) * (CANVAS - 1) / (TABLET_MAX - TABLET_MIN))


def render(points, destination):
    image = Image.new("RGB", (CANVAS, CANVAS), "white")
    draw = ImageDraw.Draw(image)
    for first, second in zip(points, points[1:]):
        if first[2] and second[2]:
            draw.line((pixel(first[0]), pixel(first[1]), pixel(second[0]), pixel(second[1])),
                      fill=(30, 30, 30), width=2)
    image.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, label, cohort in sources():
        points = read_static_points(path)
        if len(points) < 10:
            raise ValueError(f"Too few static-spiral points: {path}")
        subject = f"UCI_{cohort}_{path.stem.upper()}"
        destination = OUTPUT / f"{subject}.png"
        render(points, destination)
        rows.append({"path": destination, "label": label, "group": subject,
                     "source": "UCI", "cohort": cohort, "points": len(points)})
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rendered {len(rows)} subjects to {OUTPUT}")


if __name__ == "__main__":
    main()
