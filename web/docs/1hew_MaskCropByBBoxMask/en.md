# Mask Crop by BBox Mask - Crop Mask Using Bounding Box

**Node Purpose:** `Mask Crop by BBox Mask` crops a mask to the bounding box region derived from another `bbox_mask`. Handles batch cycling, progress reporting, and pads output masks to uniform size for stacking when necessary.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Source mask batch; 2D inputs are expanded to `[B,H,W]`. |
| `bbox_mask` | - | MASK | - | - | Mask defining the crop bounding box; 2D inputs are expanded similarly.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `cropped_mask` | MASK | Cropped mask batch; padded to same size across batch if needed.

## Features

- Bounding box detection: threshold-based bbox from `bbox_mask` (`>10` on `[0..255]`).
- Batch cycling: `mask[b]` uses `bbox_mask[b % bbox_batch]` to align differing batch sizes.
- Progress events: emits progress via `PromptServer` with unique node id.
- Safe crop: clamps bbox within mask bounds, crops via PIL, returns float `[0,1]`.
- Uniform stacking: pads masks to max size across the batch when sizes differ.

## Typical Usage

- ROI extraction: limit the area of interest in a mask using a previously computed bbox.
- Paired workflows: create bbox via image crop node, then refine mask regions using this node.

## Notes & Tips

- When no non-zero region exists in `bbox_mask`, the original `mask` item is returned unmodified.
- The bbox threshold (`>10`) avoids speckle noise; adjust upstream masks accordingly for robust detection.