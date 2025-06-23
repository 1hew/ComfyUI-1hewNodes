# Mask BBox Mask Crop

**Node Function:** The `Mask BBox Mask Crop` node crops masks in batches based on bounding box mask information, supporting multi-image processing and precise region extraction functionality.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `mask` | Required | MASK | - | - | Input mask to be cropped |
| `bbox_mask` | Required | MASK | - | - | Bounding box mask containing region information for cropping |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_mask` | MASK | Cropped mask based on bounding box regions |