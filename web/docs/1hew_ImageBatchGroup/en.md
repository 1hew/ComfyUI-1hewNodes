# Image Batch Group

**Node Function:** The `Image Batch Group` node is designed to split image batches into smaller groups with configurable batch sizes, overlap handling, and intelligent padding strategies. It provides flexible batch processing capabilities for sequential image workflows in ComfyUI, supporting various last batch handling modes and color-based padding.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image batch to be split into groups |
| `batch_size` | Required | INT | 81 | 1-1024 | Size of each output batch, step: 4 |
| `overlap` | Required | INT | 0 | 0-1024 | Number of overlapping frames between consecutive batches, step: 1 |
| `last_batch_mode` | Required | COMBO[STRING] | backtrack_last | drop_incomplete, keep_remaining, backtrack_last, fill_color | Strategy for handling the final batch |
| `color` | Optional | STRING | "1.0" | - | Color specification for padding images (supports RGB, hex, or grayscale values) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Processed image batch (may include padding images based on mode) |
| `group_total` | INT | Total number of groups created |
| `start_index` | INT | Starting index for each batch (list output) |
| `batch_count` | INT | Number of images in each batch (list output) |
| `valid_count` | INT | Number of effective (non-overlapping) images in each batch (list output) |