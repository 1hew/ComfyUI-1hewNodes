# Image Align Change Mask

Compares an original image with an AI-edited / inpainted result, corrects small global translation, rotation, scale, and color drift, and extracts the region actually changed. It outputs the aligned image, raw/clean change masks, a change heatmap, and an overlay preview.

## Use cases

- Flux Kontext, Qwen Image, GPT Image, and similar editors drift the whole image slightly (color, texture, or a few pixels) outside the local edit;
- Re-integrate only the changed content from an edit back into the original;
- Generate the next Inpaint mask from what the model *actually* changed.

> Both images must represent the same canvas and subject, and their width and height must match exactly. Whole-image restyling cannot be reliably separated from local edits.

## Inputs

| Input | Description |
|---|---|
| `edit_image` | AI-edited result. Its dimensions must match `org_image` exactly. |
| `org_image` | Pre-edit reference image. |
| `mode` | `auto` (default) evaluates none / translation / similarity. `none` disables alignment, `translation` allows only shift, `similarity` allows constrained shift, rotation, and isotropic scale. Explicit modes still fall back to `none` when they do not improve the background loss. |
| `max_offset` | Maximum accepted translation, normally 4–8 px. |
| `max_rotation` | Maximum similarity rotation; keep near 1–2 degrees. |
| `max_scale` | Maximum isotropic scale deviation; `0.03` permits 0.97–1.03. |
| `min_align_score` | Minimum candidate confidence; falls back safely when unmet. |
| `color_compensation` | Default on. Fits per-channel gain/bias only from stable unchanged pixels. Disable when a global color grade should count as a change. |
| `sensitivity` | Change sensitivity. **Lower is more sensitive** (larger mask), higher is cleaner. Default `8.0`. |
| `min_component_area` | Minimum Clean Mask component area in pixels. Lower it to keep small text, lashes, or jewelry. |
| `expand` | Pixels to grow the Clean Mask outward. |
| `feather` | Softens the Clean Mask edge; use `0` for a binary mask. |

## Outputs

| Output | Description |
|---|---|
| `align_image` | Edited image registered to the original coordinate system. Use this (not the raw `edit_image`) for later compositing. |
| `clean_mask` | Denoised, area-filtered, expanded, and feathered soft mask for inpaint / compositing. |
| `raw_mask` | High-recall binary change mask for inspecting what actually changed. |
| `change_score` | Continuous 0–1 change heatmap for diagnostics and custom thresholds. |
| `overlay_image` | Original image with the Clean Mask tinted red, for preview. |

## Recommended graph

```text
Original ────────┐
                 ├─ Image Align Change Mask ─ clean_mask ─ Inpaint / Composite
Edited result ───┘                            ├─ align_image ─┐
                                              └─ overlay_image ─ Preview
```

### Composite changes back into the original

```text
org_image ──────────────┐
                        ├─ Image Align Change Mask
edit_image ─────────────┘
                           ├─ align_image
                           └─ clean_mask

org_image ────────────────┐
align_image ───────────────┼─ Composite Masked → final image
clean_mask ────────────────┘
```

## Presets

### A: general local edit

```text
mode: auto
max_offset: 8
max_rotation: 2
max_scale: 0.03
min_align_score: 0.80
color_compensation: true
sensitivity: 8.0
min_component_area: 32
expand: 16
feather: 8
```

### B: small text / facial features / jewelry

```text
sensitivity: 5.0
min_component_area: 4
expand: 4
feather: 1
```

### C: keep only obvious large-object edits

```text
sensitivity: 12.0
min_component_area: 200
expand: 12
feather: 6
```

## Troubleshooting

1. Inspect `align_image` and `overlay_image` first to confirm a real edit was not wrongly registered;
2. If object outlines are selected as change, check for scale/rotation and use `auto` or `similarity`;
3. If the mask contains too much model noise, raise `sensitivity` and `min_component_area`;
4. If small real edits are missed, lower `sensitivity` and `min_component_area`;
5. If a global color grade should also count as a change, disable `color_compensation`.

## Algorithm and limits

Auto mode compares identity, phase/ECC translation, and feature-RANSAC/ECC constrained similarity candidates. More complex geometry is accepted only when it materially lowers a trimmed gradient residual while satisfying displacement, rotation, scale, and confidence limits. The selected result then uses validated stable-background color compensation, Lab/gradient residual fusion, hysteresis thresholding, and conservative cleanup.

The node deliberately does not apply dense optical-flow warping, which could remove a real generated edit. For large recomposition, viewpoint change, occlusion, textureless scenes, or whole-image restyles, review `change_score`, `overlay_image`, and `align_image` manually.
