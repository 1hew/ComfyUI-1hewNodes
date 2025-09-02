# Image Solid Qwen Image

**Node Function:** The `Image Solid Qwen Image` node generates solid color images based on QwenImage dimension presets, supporting multiple aspect ratios optimized for Qwen vision model workflows.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `preset_size` | - | COMBO[STRING] | 1328×1328 (1:1) | QwenImage preset options | Preset size selection optimized for Qwen vision model, includes common ratios from 9:16 to 16:9 |
| `color` | - | STRING | 1.0 | Color formats | Image color, supports grayscale (0.0-1.0), hex (#RRGGBB), and RGB (R,G,B) formats |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Generated solid color image |
| `mask` | MASK | Corresponding mask image |