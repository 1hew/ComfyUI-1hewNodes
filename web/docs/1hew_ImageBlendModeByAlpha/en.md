# Image Blend Mode by Alpha

**Node Function:** The `Image Blend Mode by Alpha` node blends an overlay image onto a base image using common graphic blend modes with adjustable opacity. Supports optional masks, RGBA-to-RGB flattening, size alignment and batch cycling.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `overlay_image` | Required | IMAGE | - | - | Overlay image batch |
| `base_image` | Required | IMAGE | - | - | Base image batch |
| `blend_mode` | Required | COMBO[STRING] | normal | mode list | Blend mode: `normal`, `dissolve`, `darken`, `multiply`, `color_burn`, `linear_burn`, `add`, `lighten`, `screen`, `color_dodge`, `linear_dodge`, `overlay`, `soft_light`, `hard_light`, `linear_light`, `vivid_light`, `pin_light`, `hard_mix`, `difference`, `exclusion`, `subtract`, `divide`, `hue`, `saturation`, `color`, `luminosity` |
| `opacity` | Required | FLOAT | 1.0 | 0.0–1.0 | Blend opacity strength |
| `overlay_mask` | Optional | MASK | - | - | Optional mask, applied on the blended result |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Invert mask (1-mask) before applying |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Blended image using the selected mode and opacity |
