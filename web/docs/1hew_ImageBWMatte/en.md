# Image BW Matte - Simplified cutout mask

**Node Purpose:** `Image BW Matte` converts an input image into a cutout mask using a fixed `auto + soft` pipeline, outputs only `mask`, and keeps only the 3 most useful controls.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image, supports single image or batch |
| `gamma` | - | FLOAT | `1.0` | `0.1 ~ 5.0` | Midtone curve; values below `1` preserve glow and semi-transparent edges more easily |
| `shrink_radius` | - | INT | `0` | `0 ~ 128` | Inward edge shrink radius in pixels; useful for trimming halos or bright fringes |
| `blur_radius` | - | FLOAT | `0.0` | `0.0 ~ 32.0` | Feather radius applied to the final mask |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Final mask ready for cutout, crop, or alpha-compositing workflows |

## Fixed Pipeline

- Auto source: prefers an existing alpha channel; without usable alpha it uses background difference when the border looks stable, otherwise it falls back to bright-value extraction.
- Fixed soft matte: preserves continuous grayscale instead of forcing a hard binary threshold, which is better for glow, haze, and semi-transparent edges.
- Fixed levels: uses built-in default black/white point stretching and no longer exposes `levels_low` / `levels_high`.

## Recommended Settings

- Typical glow-on-black assets: `gamma=0.9~1.0`, `shrink_radius=0~1`, `blur_radius=0~1`
- Visible bright fringe around the subject: start with `shrink_radius=1~3`
- Mask feels too harsh: keep `shrink_radius` small and add a little `blur_radius`
- Mask looks too fat: try `shrink_radius` before pushing `gamma` too far

## Why Keep an Inward Shrink Control

- Yes, this control is worth keeping.
- In real cutout workflows, edges often keep a halo, glow rim, or background contamination, especially with emissive subjects on dark backgrounds.
- `shrink_radius` lightly contracts the mask inward; in most cases `1~2 px` is enough.
- Large values can eat fine details, so small adjustments are recommended.

## Notes & Tips

- This is a lightweight, model-free classic matte node for solid-background, high-contrast, or emissive assets.
- For complex photographic backgrounds, prefer model-based nodes such as `Detect Remove BG`.
