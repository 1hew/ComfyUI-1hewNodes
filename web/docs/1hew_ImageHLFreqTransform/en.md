# Image HL Freq Transform - Masked Detail Transfer

**Node Purpose:** `Image HL Freq Transform` transfers detail from a `detail_image` onto a `generate_image` using `rgb`, `hsv`, or `igbi` separation methods, with optional `detail_mask` to control where the detail applies. Outputs the result and the intermediate high/low layers.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `generate_image` | - | IMAGE | - | - | Base image to receive detail. |
| `detail_image` | - | IMAGE | - | - | Source image providing detail. |
| `detail_mask` | optional | MASK | - | - | Mask controlling per-pixel detail transfer; full-white when absent. |
| `method` | - | COMBO | `igbi` | `rgb` / `hsv` / `igbi` | Separation method for transfer. |
| `blur_radius` | - | FLOAT | 10.0 | 0.0–100.0 | Gaussian blur radius; ensured odd and ≥3.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Result image with transferred detail. |
| `high_freq` | IMAGE | Intermediate high-frequency layer (after mixing). |
| `low_freq` | IMAGE | Intermediate low-frequency layer.

## Features

- Batch expansion: repeats smaller batches across `generate_image`, `detail_image`, and `detail_mask` to match the largest.
- Mask semantics: when absent, a full-white 3-channel mask is used; otherwise it is broadcast to 3 channels.
- Methods:
- `igbi` transfer: builds high/low via invert+blur mixes from both images, blends highs by the mask, then mixes `0.65*high + 0.35*low` and applies levels.
- `rgb/hsv` transfer: separates high/low from both images, blends highs by the mask, and recombines via linear light or HSV.
- Odd radius: blur radius normalized to an odd integer.

## Typical Usage

- Texture from reference: bring fine texture from a reference into a generated base while preserving base tone in low-frequency.
- Masked retouch: apply detail only in selected regions with `detail_mask` (e.g., skin pores, fabric).
- Method selection: choose `hsv` for brightness-only transfer, `rgb` for direct contrast transfer, `igbi` for level-managed stylization.

## Notes & Tips

- Inputs are converted to NumPy for OpenCV operations; outputs are in `[0,1]` float32.
- Large `blur_radius` widens the low-frequency envelope and softens high-frequency extraction.