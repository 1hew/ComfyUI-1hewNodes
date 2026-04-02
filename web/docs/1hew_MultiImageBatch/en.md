# Multi Image Batch - Align Images to Reference Size

**Node Purpose:** `Multi Image Batch` aligns multiple images to the size of the first image using three fit modes: `crop`, `pad`, and `stretch`. It supports advanced padding color strategies and concatenates all aligned images into one batch. If any input has alpha, all images are processed as RGBA and the output automatically keeps alpha.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `fit` | - | COMBO | `pad` | `crop` / `pad` / `stretch` | Alignment mode. |
| `pad_color` | - | STRING | `1.0` | color name/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Padding color or strategy. |
| `image_1` | - | IMAGE | - | - | First image; defines reference size. |
| `image_2…image_N` | optional | IMAGE | - | - | Additional images recognized by numeric suffix ordering.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch of aligned images; clamped to `[0,1]` float.

## Features

- Reference size: uses height/width of `image_1` as target.
- Channel rule: if any input contains alpha, every image is expanded to RGBA and the output preserves alpha automatically.
- Fit modes:
- `stretch`: bicubic resize directly to target.
- `crop`: scale to cover target, then center-crop.
- `pad`: scale to fit inside target, then pad with `pad_color`.
- Advanced padding: `extend` replicate, `mirror` reflect (segmented), `edge` per-side averages, `average` global average, or explicit colors.
- Ordering: collects `image_*` inputs by numeric suffix to preserve sequence.

## Typical Usage

- Unify image batches: standardize sizes before stacking or model ingestion.
- Safe padding: choose `extend/mirror` for natural borders; `edge` harmonizes colors; `average` for uniform tone.
- Cover vs fit: pick `crop` for full coverage, `pad` to preserve entire content.

## Notes & Tips

- The first image determines target height/width; ensure it matches desired output.
- When the output is RGBA, explicit padding colors are automatically extended with opaque alpha.
- Padding color parser supports grayscale floats, `R,G,B` in `0..1` or `0..255`, `#hex`, and common names.