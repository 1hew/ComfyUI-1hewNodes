# Image Blend Modes By CSS

**Node Function:** The `Image Blend Modes By CSS` node implements CSS-standard blending modes for image composition, providing web-compatible blending effects with precise color management.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `base_image` | Required | IMAGE | - | - | Base layer image |
| `overlay_image` | Required | IMAGE | - | - | Overlay layer image |
| `blend_mode` | - | COMBO[STRING] | normal | CSS blend modes | CSS-standard blending mode |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity of the overlay layer |
| `overlay_mask` | Optional | MASK | - | - | Optional mask for selective blending |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert the overlay mask |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | CSS-blended result image |