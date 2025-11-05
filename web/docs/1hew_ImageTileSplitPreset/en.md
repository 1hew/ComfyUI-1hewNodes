# Image Tile Split Preset

**Node Function:** The `Image Tile Split Preset` node intelligently splits large images into multiple tiles using predefined resolution presets. It offers automatic size selection or manual preset selection, optimizing tile efficiency and providing comprehensive metadata for seamless tile processing workflows.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be split into tiles |
| `overlap_amount` | Required | FLOAT | 0.05 | 0.0-512.0 | Overlap amount: ≤1.0 for ratio mode, >1.0 for pixel mode |
| `tile_preset_size` | Required | COMBO[STRING] | auto | auto + preset list | Tile size selection: auto or predefined resolutions |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `tiles` | IMAGE | Split image tile batch with consistent predefined dimensions |
| `tiles_meta` | DICT | Comprehensive tile metadata including grid layout, efficiency metrics, and reconstruction information |