# Mask BBox Crop

**Node Function:** The `Mask BBox Crop` node crops masks in batches based on bounding box information, supporting multi-image processing and precise region extraction functionality.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `mask` | Required | MASK | - | - | Input mask to be cropped |
| `bbox_meta` | Required | DICT | - | - | Bounding box metadata containing crop position and size information |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_mask` | MASK | Cropped mask |