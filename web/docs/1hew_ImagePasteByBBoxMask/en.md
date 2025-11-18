# Image Paste By BBox Mask - Transform-and-Paste by Bounding Mask

**Node Purpose:** `Image Paste By BBox Mask` fits, transforms, and pastes a processed image into a target region defined by a `bbox_mask` on the base image. Supports optional `paste_mask`, position offsets, scaling, rotation, and opacity.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `paste_image` | - | IMAGE | - | - | Processed image to paste; RGBA alpha is respected. |
| `base_image` | - | IMAGE | - | - | Target image onto which the paste occurs. |
| `bbox_mask` | - | MASK | - | - | Bounding box area where the paste should be fitted and centered. |
| `paste_mask` | optional | MASK | - | - | Optional mask controlling per-pixel transparency of the paste. |
| `position_x` | - | INT | 0 | -4096–4096 | Horizontal offset from the bbox center. |
| `position_y` | - | INT | 0 | -4096–4096 | Vertical offset from the bbox center. |
| `scale` | - | FLOAT | 1.0 | 0.1–10.0 | Scale factor relative to the fitted size inside the bbox. |
| `rotation` | - | FLOAT | 0.0 | -3600.0–3600.0 | Rotation in degrees; positive values rotate clockwise. |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Global opacity; also scales `paste_mask`/alpha if present. |
| `apply_paste_mask` | - | BOOLEAN | false | - | If true, crops `paste_image` and `paste_mask` by the non-empty mask region before transform/paste. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Base image with the transformed paste applied. |
| `mask` | MASK | Output mask marking the pasted area in the base image.

## Features

- BBox detection: extracts bbox from `bbox_mask` to compute fit and center.
- Fit and transform: preserves aspect by fitting to bbox, then applies `scale`, `rotation`, and offsets.
- Alpha and opacity: handles RGBA alpha; scales mask/alpha by `opacity`.
- Safe placement: clamps paste to base bounds and crops out-of-bounds regions; returns blank mask if no overlap.
- Batch robustness: cycles batches and stacks device-clamped outputs.

## Typical Usage

- Restore processed crops: pair with `Image Crop With BBox Mask` to paste edited regions back to their original locations.
- Guided placement: use `paste_mask` for soft edges; adjust `position_x`/`position_y` to nudge placement.
- Creative transforms: combine `scale` and `rotation` for composite assemblies and retouching workflows.

## Notes & Tips

- When `apply_paste_mask` is true, the paste and mask are cropped to the non-empty mask bbox before fitting.
- Positive `rotation` rotates clockwise; the internal implementation accounts for PIL’s rotation direction.
- If bbox is empty, the node returns the base image unchanged and an all-black output mask.