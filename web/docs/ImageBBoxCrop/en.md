# Image BBox Crop

**Node Function:** The `Image BBox Crop` node crops images in batches based on bounding box information, supporting multi-image processing and size adjustment, commonly used for region extraction after object detection.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be cropped |
| `bbox_meta` | Required | DICT | - | - | Bounding box metadata containing crop position and size information |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Image cropped according to bounding box |