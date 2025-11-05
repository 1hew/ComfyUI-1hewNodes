# Video Cut Group - Video Scene Cut Detection

**Node Function:** The `Video Cut Group` node detects scene transitions in video sequences by analyzing similarity between adjacent frames. It supports both fast and precise modes for identifying hard cuts, commonly used for video segmentation and scene analysis.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input video frame sequence |
| `threshold_base` | - | FLOAT | 0.8 | 0.0-1.0 | Base threshold for frame similarity judgment |
| `threshold_range` | - | FLOAT | 0.05 | 0.01-0.2 | Threshold range for multi-threshold detection |
| `threshold_count` | - | INT | 2 | 1-10 | Number of thresholds for multi-threshold detection |
| `kernel` | - | STRING | "3, 7, 11" | Kernel config | Blur kernel size configuration, comma-separated odd values |
| `min_frame_count` | - | INT | 10 | 1-1000 | Minimum frame count per segment |
| `max_frame_count` | - | INT | 0 | 0-10000 | Maximum frame count per segment, 0 means unlimited |
| `fast` | - | BOOLEAN | False | True/False | Fast mode: True for quick detection, False for precise detection |
| `add_frame` | - | STRING | "" | Frame indices | Manually add cut points, comma-separated frame indices |
| `delete_frame` | - | STRING | "" | Frame indices | Manually remove cut points, comma-separated frame indices |

## Outputs

| Output | Data Type | Description |
|--------|-----------|-------------|
| `image` | IMAGE | Grouped image batches |
| `group_total` | INT | Total number of groups |
| `start_index` | INT | Starting index of each group (list) |
| `batch_count` | INT | Frame count of each group (list) |
