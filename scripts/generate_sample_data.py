"""Generate a synthetic, annotated car-damage **training set** (VIA 1.x format).

Draws car-like images with dent / scratch / structural damage regions under
``data/raw/`` so ``preprocess`` + ``train`` can run without a real dataset.
Everything is drawn with OpenCV — it is a *stand-in* for real data; replace
``data/raw/`` with real annotated crash photos for meaningful results.

For real photos to upload into the app, use ``scripts/download_examples.py``.

Usage:
    python scripts/generate_sample_data.py [--num-images 24] [--out data/raw]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["dent", "scratch", "structural"]

# Simple BGR car-body palette to vary the demo images.
_BODY_COLORS = [
    (60, 60, 190), (180, 120, 40), (70, 160, 70),
    (40, 40, 40), (190, 190, 190), (150, 90, 160),
]


def _ellipse_polygon(cx, cy, rx, ry, n=16, jitter=0.0, rng=None):
    """Return an ellipse-like closed polygon (optionally jittered/jagged)."""
    points = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        scale = 1.0 + (float(rng.uniform(-jitter, jitter)) if rng is not None and jitter else 0.0)
        points.append((int(cx + rx * math.cos(ang) * scale), int(cy + ry * math.sin(ang) * scale)))
    return points


def _scratch_polygon(cx, cy, length, width, angle):
    """Return a thin rotated quad approximating a scratch."""
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = -dy, dx
    hl, hw = length / 2.0, width / 2.0
    corners = [
        (cx - dx * hl - px * hw, cy - dy * hl - py * hw),
        (cx + dx * hl - px * hw, cy + dy * hl - py * hw),
        (cx + dx * hl + px * hw, cy + dy * hl + py * hw),
        (cx - dx * hl + px * hw, cy - dy * hl + py * hw),
    ]
    return [(int(x), int(y)) for x, y in corners]


def _draw_car(canvas, rng):
    """Draw a simple car over a road/sky background; return (body_rect, color)."""
    height, width = canvas.shape[:2]
    canvas[:, :] = (235, 225, 205)                                     # sky
    cv2.rectangle(canvas, (0, int(height * 0.72)), (width, height), (95, 95, 95), -1)  # road

    color = tuple(int(c) for c in _BODY_COLORS[int(rng.integers(len(_BODY_COLORS)))])
    bx, by = int(width * 0.12), int(height * 0.44)
    bw, bh = int(width * 0.76), int(height * 0.26)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, -1)

    cabin = np.array([
        [bx + int(bw * 0.24), by],
        [bx + int(bw * 0.36), by - int(bh * 0.75)],
        [bx + int(bw * 0.70), by - int(bh * 0.75)],
        [bx + int(bw * 0.80), by],
    ], np.int32)
    cv2.fillPoly(canvas, [cabin], color)
    window = np.array([
        [bx + int(bw * 0.31), by - 2],
        [bx + int(bw * 0.40), by - int(bh * 0.58)],
        [bx + int(bw * 0.66), by - int(bh * 0.58)],
        [bx + int(bw * 0.71), by - 2],
    ], np.int32)
    cv2.fillPoly(canvas, [window], (55, 45, 40))

    wheel_r = int(bh * 0.42)
    for wx in (bx + int(bw * 0.24), bx + int(bw * 0.76)):
        cv2.circle(canvas, (wx, by + bh), wheel_r, (25, 25, 25), -1)
        cv2.circle(canvas, (wx, by + bh), int(wheel_r * 0.45), (120, 120, 120), -1)
    cv2.circle(canvas, (bx + bw - 8, by + int(bh * 0.30)), int(bh * 0.12), (200, 240, 255), -1)
    return (bx, by, bw, bh), color


def _damage_polygon(rng, body_rect, damage_type):
    """Return the polygon points for a damage region within the car body."""
    bx, by, bw, bh = body_rect
    cx = int(rng.integers(bx + int(bw * 0.12), bx + int(bw * 0.88)))
    cy = int(rng.integers(by + int(bh * 0.22), by + int(bh * 0.80)))
    if damage_type == "dent":
        return _ellipse_polygon(cx, cy, int(bw * 0.06), int(bh * 0.26), n=16, jitter=0.07, rng=rng)
    if damage_type == "scratch":
        length = int(rng.integers(int(bw * 0.18), int(bw * 0.35)))
        return _scratch_polygon(cx, cy, length, int(rng.integers(4, 10)), float(rng.uniform(-0.6, 0.6)))
    return _ellipse_polygon(cx, cy, int(bw * 0.10), int(bh * 0.34), n=11, jitter=0.45, rng=rng)


def _draw_damage(canvas, points, damage_type, body_color, rng):
    """Render the damage so it visually matches its annotation polygon."""
    poly = np.array(points, np.int32)
    if damage_type == "dent":
        cv2.fillPoly(canvas, [poly], tuple(int(c * 0.55) for c in body_color))
        x, y, w, h = cv2.boundingRect(poly)
        cv2.ellipse(canvas, (x + w // 2, y + h // 2),
                    (max(2, w // 3), max(2, h // 3)), 0, 0, 360,
                    tuple(int(c * 0.35) for c in body_color), -1)
    elif damage_type == "scratch":
        cv2.fillPoly(canvas, [poly], (225, 225, 230))
        cv2.polylines(canvas, [poly], True, (150, 150, 160), 1)
    else:  # structural
        cv2.fillPoly(canvas, [poly], (35, 35, 42))
        cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
        for _ in range(4):
            ex = cx + int(rng.integers(-30, 30))
            ey = cy + int(rng.integers(-30, 30))
            cv2.line(canvas, (cx, cy), (ex, ey), (15, 15, 18), 1)


def _render_scene(rng, damage_type, size=(640, 480)):
    """Render one car scene; return ``(image, polygon_points)``."""
    width, height = size
    canvas = np.empty((height, width, 3), np.uint8)
    body_rect, color = _draw_car(canvas, rng)
    points = _damage_polygon(rng, body_rect, damage_type)
    _draw_damage(canvas, points, damage_type, color, rng)
    return canvas, points


def _write_via(out_dir: Path, records: list[tuple[str, int, list, str]]) -> Path:
    """Write a VIA 1.x ``via_region_data.json`` for the given image records."""
    via: dict[str, dict] = {}
    for filename, size, points, damage in records:
        via[f"{filename}{size}"] = {
            "filename": filename,
            "size": size,
            "regions": {
                "0": {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [int(x) for x, _ in points],
                        "all_points_y": [int(y) for _, y in points],
                    },
                    "region_attributes": {"damage": damage},
                }
            },
            "file_attributes": {},
        }
    annotations = out_dir / "via_region_data.json"
    annotations.write_text(json.dumps(via, indent=2), encoding="utf-8")
    return annotations


def generate_dataset(
    out_dir: Path, num_images: int, seed: int, annotated: bool, prefix: str = "car"
) -> Path | None:
    """Render ``num_images`` car scenes into ``out_dir``.

    Args:
        out_dir: Destination directory (an ``images`` subfolder is used when
            ``annotated`` is True, otherwise images are written directly).
        num_images: How many images to render.
        seed: Random seed.
        annotated: When True, also write a VIA ``via_region_data.json``.
        prefix: Filename prefix.

    Returns:
        Path to the annotations file when ``annotated``, else ``None``.
    """
    rng = np.random.default_rng(seed)
    images_dir = out_dir / "images" if annotated else out_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, int, list, str]] = []
    for i in range(num_images):
        damage = CLASSES[i % len(CLASSES)]
        image, points = _render_scene(rng, damage)
        filename = f"{prefix}_{damage}_{i:03d}.jpg"
        cv2.imwrite(str(images_dir / filename), image)
        records.append((filename, (images_dir / filename).stat().st_size, points, damage))

    if annotated:
        return _write_via(out_dir, records)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic car-damage training set.")
    parser.add_argument("--num-images", type=int, default=24, help="Training images to create.")
    parser.add_argument("--out", default="data/raw", help="Output dir for the training set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    out = Path(args.out)
    annotations = generate_dataset(out, args.num_images, args.seed, annotated=True)

    print(f"Training set : {args.num_images} images + {annotations}")
    print("\nNext steps:")
    print(f"  collision-vision preprocess --format via --annotations {annotations} "
          f"--images {out / 'images'} --attribute-key damage")
    print("  collision-vision train --config config.yaml")
    print("  python scripts/download_examples.py   # real photos to upload in the app")


if __name__ == "__main__":
    main()
