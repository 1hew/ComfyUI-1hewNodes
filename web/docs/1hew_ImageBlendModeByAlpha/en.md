# Image Blend Mode by Alpha - Alpha-driven Layer Blending

**Node Purpose:** `Image Blend Mode by Alpha` blends an overlay image onto a base image using a wide set of professional blend modes and an overall `opacity`. Supports optional per-pixel `overlay_mask`, RGBA-to-RGB normalization, size alignment, and robust batch handling.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `overlay_image` | - | IMAGE | - | - | Overlay image batch placed over `base_image`. |
| `base_image` | - | IMAGE | - | - | Base image batch. |
| `overlay_mask` | optional | MASK | - | - | Optional mask; modulates where the blended result replaces the base. |
| `blend_mode` | - | COMBO | `normal` | `normal` / `dissolve` / `darken` / `multiply` / `color_burn` / `linear_burn` / `add` / `lighten` / `screen` / `color_dodge` / `linear_dodge` / `overlay` / `soft_light` / `hard_light` / `linear_light` / `vivid_light` / `pin_light` / `hard_mix` / `difference` / `exclusion` / `subtract` / `divide` / `hue` / `saturation` / `color` / `luminosity` | Blend algorithm to apply. |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Global blend strength; 0 leaves base unchanged, 1 fully applies overlay blend. |
| `invert_mask` | - | BOOLEAN | false | - | Invert `overlay_mask` before application. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch of blended images.

## Features

- Professional blend set: normal, dissolve, darken/lighten, multiply/screen, burn/dodge (color/linear), add/subtract/divide, overlay/soft/hard/linear/vivid/pin/hard mix, difference/exclusion, plus HSL-based hue/saturation/color/luminosity.
- RGBA normalization: flattens alpha against white to ensure consistent RGB blending.
- Size alignment: overlay is resized to match base dimensions when needed.
- Masked blending: optional `overlay_mask` replaces base with blended result only where mask>0; `invert_mask` flips selection.
- Batch robustness: mismatched batch sizes are expanded by repeating smaller batches; mask batches loop over images.
- Device safety: tensors are kept on the same device throughout operations.

## Typical Usage

- Shading and lighting: use `multiply` for shading, `screen` for brightening.
- Contrast and punch: `overlay` / `hard_light` for strong contrast boosts.
- Color harmonization: `hue` / `saturation` / `color` / `luminosity` to mix color properties without changing structure.
- Region-limited blending: provide `overlay_mask` to apply effects only in selected areas; use `invert_mask` when needed.

## Notes & Tips

- `opacity` scales the blended result against the base; combine with masks for precise control.
- `divide` guards against division by zero internally; results are clamped to valid ranges.
- `dissolve` introduces randomness; higher `opacity` increases the proportion of overlay pixels.
- RGBA inputs are flattened to white; for different background assumptions, pre-process images accordingly.