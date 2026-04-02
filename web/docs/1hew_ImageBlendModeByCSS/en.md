# Image Blend Mode by CSS - CSS-Compatible Image Blending

**Node Purpose:** `Image Blend Mode by CSS` reproduces CSS blending semantics between a base image and an overlay image, with explicit `blend_mode`, overall `blend_percentage`, and optional `overlay_mask` control. Includes HSL-based modes for `hue`, `saturation`, `color`, and `luminosity`, with RGBA-aware blending.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `overlay_image` | - | IMAGE | - | - | Overlay image batch placed over `base_image`. |
| `base_image` | - | IMAGE | - | - | Base image batch. |
| `blend_mode` | - | COMBO | `normal` | `normal` / `multiply` / `screen` / `overlay` / `darken` / `lighten` / `color_dodge` / `color_burn` / `hard_light` / `soft_light` / `difference` / `exclusion` / `hue` / `saturation` / `color` / `luminosity` | CSS blend algorithm to apply. |
| `blend_percentage` | - | FLOAT | 100.0 | 0–100 | Global opacity (percent) of the blended result. |
| `overlay_mask` | optional | MASK | - | - | Optional mask; restricts blended replacement to mask>0 areas. |
| `invert_mask` | - | BOOLEAN | false | - | Invert `overlay_mask` before application. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch of blended images.

## Features

- CSS-accurate formulas: match typical CSS semantics for multiply, screen, overlay, darken, lighten, color dodge/burn, hard/soft light, difference, exclusion.
- HSL modes: `hue`, `saturation`, `color`, `luminosity` implemented via RGB↔HSL conversion.
- Opacity control: `blend_percentage` (0–100) linearly mixes base and blended results.
- RGBA support: both `overlay_image` and `base_image` may contain alpha, and the overlay alpha participates in blending.
- Output rule: if `base_image` has alpha, the node outputs RGBA; otherwise it outputs RGB.
- Size alignment and masking: overlay is resized to base; optional mask is resized and broadcast to RGB.
- Robust batching: mismatched base/overlay/mask batch sizes loop across the maximum batch size.

## Typical Usage

- Photo toning: `multiply` for shadows, `screen` for highlights, `overlay` to boost contrast.
- Color-only mixing: use HSL modes to transfer hue/saturation/luminosity while keeping base structure.
- Region-specific blending: feed `overlay_mask` to limit blending; enable `invert_mask` to flip selection.
- Strength control: tune `blend_percentage` to match the desired effect intensity.

## Notes & Tips

- When overlay size differs, it is resized with Lanczos before blending.
- If you want the final result to keep transparency, feed `base_image` as RGBA.
- `color_dodge` and `color_burn` handle boundary cases to avoid infinities and negatives.