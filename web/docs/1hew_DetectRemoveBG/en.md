# Detect Remove BG - Multi-backend Background Removal

**Node Purpose:** `Detect Remove BG` provides a unified background-removal entry with multiple backends (`RMBG-1.4`, `RMBG-2.0`, `birefnet`, `Inspyrenet`, etc.), and outputs both foreground image and alpha mask. It can return transparent RGBA or composited white/black backgrounds.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | required | IMAGE | - | - | Input image batch. |
| `model` | - | COMBO | `RMBG-1.4` | `none` / `RMBG-1.4` / `RMBG-2.0` / `birefnet-general` / `birefnet-general-lite` / `Inspyrenet` | Segmentation backend selector. |
| `add_background` | - | COMBO | `alpha` | `alpha` / `white` / `black` | Output background mode. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Foreground result: RGBA in `alpha` mode, RGB composite in `white/black` mode. |
| `mask` | MASK | Float alpha mask in the 0~1 range. |

## Features

- Unified interface across multiple model backends.
- Automatic model preparation: some backends are downloaded to `models/rembg` and cached.
- Classic fallback: `model=none` uses color-difference based alpha estimation.
- Optional background composition for direct downstream usage.

## Typical Usage

- General background removal: `model=RMBG-1.4`, `add_background=alpha`.
- White-background export workflow: set `add_background=white`.
- A/B quality checks: run the same input with different `model` options to compare edge quality and subject retention.

## Notes & Tips

- Backend dependencies vary; missing packages will be reported in logs.
- For cleaner edges and decontamination, feed the output mask into `Detect Remove BG Refine` as a post-processing step.
