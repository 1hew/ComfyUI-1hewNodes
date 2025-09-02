# Image Solid Flux Kontext

**Node Function:** The `Image Solid Flux Kontext` node generates solid color images based on Flux Kontext dimension presets, supporting multiple aspect ratios optimized for Flux model workflows.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `preset_size` | - | COMBO[STRING] | 672×1568 [1:2.33] (3:7) | Flux Kontext preset options | Preset size selection optimized for Flux model, includes ratios from 1:2.33 to 2.33:1 |
| `color` | - | STRING | 1.0 | Color formats | Image color, supports grayscale (0.0-1.0), hex (#RRGGBB), and RGB (R,G,B) formats |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Generated solid color image |
| `mask` | MASK | Corresponding mask image |