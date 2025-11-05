# Image Blend Mode by CSS

**Node Function:** The `Image Blend Mode by CSS` node applies CSS blend modes implemented via the `pilgram` library to blend an overlay image with a base image. Supports mask application, blend percentage, RGBA-to-RGB flattening, and batch cycling.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `overlay_image` | Required | IMAGE | - | - | Overlay image batch |
| `base_image` | Required | IMAGE | - | - | Base image batch |
| `blend_mode` | Required | COMBO[STRING] | normal | mode list | CSS blend mode: `normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, `color_dodge`, `color_burn`, `hard_light`, `soft_light`, `difference`, `exclusion`, `hue`, `saturation`, `color`, `luminosity` |
| `blend_percentage` | Required | FLOAT | 1.0 | 0.0–1.0 | Strength of CSS blend (post-application scaling) |
| `overlay_mask` | Optional | MASK | - | - | Optional mask to spatially mix the result |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Invert mask before applying |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image blended using the selected CSS mode and percentage |
