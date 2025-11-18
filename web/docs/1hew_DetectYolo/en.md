# Detect Yolo - Object detection with masks and labels

**Node Purpose:** `Detect Yolo` performs object detection using an Ultralytics YOLO model. It annotates images with bounding boxes and optional labels, and outputs an aggregated binary mask from per-instance masks (or a bbox-based fallback when segmentation masks are unavailable).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `yolo_model` | - | COMBO | model list | .pt files | YOLO model selection; lists `.pt` files under `models/yolo`. You can also type a path relative to that directory. |
| `threshold` | - | FLOAT | 0.3 | 0.0-1.0 | Confidence threshold (`conf`). |
| `mask_index` | - | STRING | `-1` | comma list | Detection indices to include; `-1` merges all; supports Chinese/English commas. |
| `label` | - | BOOLEAN | `True` | - | Whether to draw labels next to boxes. |
| `label_size` | - | FLOAT | 1.0 | 0.1-5.0 | Label scale affecting font and box thickness. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `plot_image` | IMAGE | Annotated image with bounding boxes and optional labels. |
| `mask` | MASK | Aggregated binary mask (`B×H×W`, `[0,1]`) combining selected instances. |

## Features

- Model cache: caches loaded YOLO models by path to avoid repeated initialization.
- Mask aggregation: sums selected instance masks and clamps to `[0,1]`; falls back to bbox-filled masks if segmentation masks are absent.
- Labeling: draws `[index] class conf` with adjustable size and padded background for readability.
- Path discovery: looks for models in ComfyUI `models/yolo`, falls back to base path, or creates `models/yolo` in the current directory.
- Batch async: per-frame inference dispatched to threads with serialized access.

## Typical Usage

- Place `.pt` models under `models/yolo` (e.g., `yolov8n-seg.pt`), select from the dropdown.
- Set `threshold` for desired confidence; use `mask_index=-1` to merge all masks or specify indices like `0,1`.
- Adjust `label_size` for display; set `label=False` for clean masks-only workflows.

## Notes & Tips

- When no masks are produced by the model, a bbox-based binary mask is generated per detection.
- `mask_index` uses detection order; out-of-range indices are ignored.
- Logs are printed with concise tags to aid debugging.