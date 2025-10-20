# Image Batch Extract

**Node Function:** The `Image Batch Extract` node is used to extract specific images from a batch of images, supporting multiple extraction modes including custom indices, step intervals, and uniform distribution, commonly used for frame sampling and batch processing optimization.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input batch of images to extract from |
| `mode` | - | COMBO[STRING] | step | index, step, uniform | Extraction mode: index (custom indices), step (interval sampling), uniform (evenly distributed) |
| `index` | - | STRING | "0" | - | Custom indices string, supports comma-separated values and negative indices (e.g., "0,2,4" or "1,-1") |
| `step` | - | INT | 4 | 1-8192 | Step interval for step mode, extracts every Nth image starting from index 0 |
| `uniform` | - | INT | 4 | 0-8192 | Number of images to extract for uniform mode, evenly distributed across the batch |
| `max_keep` | - | INT | 10 | 0-8192 | Maximum number of images to keep in the result, prevents excessive memory usage |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Extracted batch of images |