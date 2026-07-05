# CollisionVision

Instance segmentation of post-accident car damage — **dents**, **scratches** and
**structural** damage — using [Ultralytics YOLOv8-Seg](https://docs.ultralytics.com/tasks/segment/).
CollisionVision converts polygon annotations from **VGG Image Annotator (VIA)** or
**COCO** into the YOLOv8-Seg label format, fine-tunes a pre-trained segmentation
model, and serves an interactive Streamlit app that overlays the predicted masks
directly on an uploaded photo.

---

## Features

- **Two annotation converters** — VIA (1.x & 2.x) and COCO polygon segmentation → YOLOv8-Seg.
- **Config-driven fine-tuning** of `yolov8s-seg.pt` via [`config.yaml`](config.yaml).
- **Clean per-instance mask extraction** (`DamageInstance`: binary mask, contour, bbox, area).
- **Streamlit app** for drag-and-drop inference with a side-by-side mask overlay.
- **CLI** (`collision-vision`) for `preprocess` / `train` / `infer`.
- **Tests, Makefile and Dockerfile** included.

## Project structure

```
collision_vision/
├── app.py                       # Streamlit inference app
├── config.yaml                  # Training hyperparameters (fine-tuning)
├── requirements.txt
├── src/collision_vision/
│   ├── config.py                # config.yaml loader
│   ├── converters/
│   │   ├── base.py              # BaseConverter + polygon normalization
│   │   ├── via_converter.py     # VIA  -> YOLOv8-Seg
│   │   └── coco_converter.py    # COCO -> YOLOv8-Seg
│   ├── split.py                 # train/val splitter
│   ├── preprocess.py            # convert -> split -> data.yaml
│   ├── train.py                 # SegmentationTrainer (Ultralytics wrapper)
│   ├── inference.py             # DamageSegmenter + DamageInstance
│   ├── visualize.py             # MaskVisualizer overlay
│   └── cli.py                   # `collision-vision` entry point
├── tests/                       # converter unit tests
├── data/{raw,processed}/        # inputs / generated dataset
└── models/                      # trained weights
```

## Installation

```bash
python -m pip install -e .          # runtime
python -m pip install -r requirements-dev.txt   # + tests
```

## Quick start (demo data)

No dataset yet? Generate a small synthetic one to exercise the whole pipeline:

```bash
python scripts/generate_sample_data.py       # synthetic training set -> data/raw/
collision-vision preprocess --format via \
  --annotations data/raw/via_region_data.json --images data/raw/images --attribute-key damage
collision-vision train --config config.yaml --epochs 5
cp runs/segment/collision_vision/weights/best.pt models/best.pt
streamlit run app.py                          # then upload an image from data/examples/
```

> The synthetic training shapes are random, so a model trained only on `data/raw/`
> will **not** perform on real photos — train on a real annotated dataset (see
> "Using a real dataset" below) for meaningful results. `data/examples/` holds
> real damaged-car photos to upload in the app.

## Using a real dataset

The demo images are synthetic. For real results you need real annotated crash
photos — two common routes:

1. **Annotate your own** with [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)
   (export `via_region_data.json`), or any tool that exports COCO segmentation,
   then run `collision-vision preprocess` as shown above.
2. **Use a public dataset** — e.g. search [Roboflow Universe](https://universe.roboflow.com/)
   for "car damage segmentation" and export as **COCO Segmentation** (polygons)
   or **YOLOv8**:

   ```bash
   collision-vision preprocess --format coco \
     --annotations path/to/_annotations.coco.json --images path/to/images
   ```

   If the export is already in YOLOv8-Seg layout (`images/`, `labels/`, a
   `data.yaml`), just point `config.yaml`'s `data:` at that `data.yaml` and skip
   preprocessing.

Aim for at least a few hundred labelled instances per class before expecting
usable accuracy.

---

## Annotation formats → YOLOv8-Seg

### The YOLOv8-Seg label format

YOLOv8 segmentation uses **one `.txt` file per image** (same basename). Each line
describes one object instance as a class id followed by a flattened list of
**polygon vertices normalized to `[0, 1]`**:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- `class_id` — zero-based (`0=dent`, `1=scratch`, `2=structural`).
- `xi = pixel_x / image_width`, `yi = pixel_y / image_height` (clipped to `[0, 1]`).

The dataset is arranged in the standard Ultralytics layout, described by a
generated `data.yaml`:

```
data/processed/
├── images/train/  images/val/
├── labels/train/  labels/val/
└── data.yaml
```

### 1. VGG Image Annotator (VIA) polygons

VIA stores polygons as parallel `all_points_x` / `all_points_y` arrays inside each
region's `shape_attributes`. Both layouts are supported automatically:

- **VIA 1.x** (`via_region_data.json`): a flat `{ "<file><size>": { ... } }` map
  whose `regions` is a **dict**.
- **VIA 2.x** project export: wrapped in `_via_img_metadata` with `regions` as a **list**.

```json
{
  "car01.jpg148923": {
    "filename": "car01.jpg",
    "regions": {
      "0": {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [120, 180, 200, 140],
          "all_points_y": [90, 85, 150, 160]
        },
        "region_attributes": { "damage": "dent" }
      }
    }
  }
}
```

The damage class comes from a `region_attributes` field. Pass `--attribute-key`
to say which key holds the label (e.g. `damage`); otherwise the first non-empty
attribute value is used. VIA does not record image dimensions, so the original
images are read from `--images` to normalize the coordinates.

```bash
collision-vision preprocess \
  --format via \
  --annotations data/raw/via_region_data.json \
  --images data/raw/images \
  --attribute-key damage \
  --output data/processed
```

> `polygon`, `polyline` and `rect` VIA shapes are supported (rectangles are
> converted to their four corners).

### 2. COCO segmentation

COCO stores polygons in each annotation's `segmentation` field as one or more
flattened `[x1, y1, x2, y2, ...]` lists, with the class in `category_id`:

```json
{
  "images":      [{ "id": 1, "file_name": "car01.jpg", "width": 1024, "height": 768 }],
  "annotations": [{ "image_id": 1, "category_id": 2, "segmentation": [[120,90, 180,85, 200,150, 140,160]] }],
  "categories":  [{ "id": 1, "name": "dent" }, { "id": 2, "name": "scratch" }, { "id": 3, "name": "structural" }]
}
```

Image dimensions are taken directly from the JSON, and COCO `category_id`s are
mapped to zero-based class indices (by category **name** when it matches your
class list, otherwise by order of appearance).

```bash
collision-vision preprocess \
  --format coco \
  --annotations data/raw/instances.json \
  --images data/raw/images \
  --output data/processed
```

> **Note:** only polygon segmentations are converted. Run-length-encoded (RLE)
> masks are detected and skipped with a warning to avoid a `pycocotools`
> dependency. Export your annotations as polygons (Roboflow: *YOLOv8* or
> *COCO Segmentation* with polygons).

---

## Training

Hyperparameters live in [`config.yaml`](config.yaml) and are tuned for
**fine-tuning** pre-trained weights (`yolov8s-seg.pt`, `AdamW`, low `lr0`, frozen
backbone, segmentation-friendly `copy_paste` augmentation).

```bash
collision-vision train --config config.yaml
# override on the fly (checkpoint, size, schedule, device):
collision-vision train --config config.yaml --model yolov8n-seg.pt --imgsz 512 --epochs 50 --batch 8 --device 0
```

Results (weights, curves, metrics) are written under `runs/segment/collision_vision/`.
Copy the best checkpoint to `models/` for inference:

```bash
cp runs/segment/collision_vision/weights/best.pt models/best.pt
```

## Inference (CLI)

```bash
collision-vision infer --weights models/best.pt --source data/raw/images --output outputs
```

Each output image is the original with translucent masks, contours and
`"<class> <confidence>"` labels drawn on top.

## Streamlit app

```bash
streamlit run app.py
```

Set the weights path in the sidebar (default `models/best.pt`), upload a photo of
a crashed car, and inspect the segmentation overlay side-by-side with the
original plus a per-instance table (damage type, confidence, mask area).

No photo handy? `data/examples/` already contains real damaged-car photos to
upload. (`python scripts/download_examples.py` can also fetch openly-licensed
crash photos from Wikimedia Commons.)

## Docker

```bash
docker build -t collision-vision .
docker run --rm -p 8501:8501 -v "$PWD/models:/app/models" collision-vision
# open http://localhost:8501
```

## Testing

```bash
pytest
```

The tests cover the VIA and COCO converters end-to-end (parsing, class mapping,
coordinate normalization and label writing) and run fully offline.

## Programmatic use

```python
import cv2
from collision_vision.inference import DamageSegmenter
from collision_vision.visualize import MaskVisualizer

segmenter = DamageSegmenter("models/best.pt", conf=0.25)
image = cv2.imread("crash.jpg")
instances = segmenter.predict(image)

for d in instances:
    print(d.class_name, round(d.confidence, 2), "area_px=", d.area_px)

overlay = MaskVisualizer().overlay(image, instances)
cv2.imwrite("crash_segmented.jpg", overlay)
```

## Acknowledgements

Inspired by earlier Mask R-CNN car-damage work by
[Priya Dwivedi (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2018/07/building-mask-r-cnn-model-detecting-damage-cars-python/)
and related repositories, re-implemented here with the modern YOLOv8-Seg stack.