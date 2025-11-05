# Image List Append

**Node Function:** The `Image List Append` node is used to append two image inputs into a list format, supporting intelligent merging of image batches, commonly used for image collection and batch processing workflows.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image_1` | Required | IMAGE | - | - | First image input, can be a single image or image batch |
| `image_2` | Required | IMAGE | - | - | Second image input, can be a single image or image batch |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image_list` | IMAGE | Merged image list |

## Features

### Use Cases
- **Image Collection**: Merge images from multiple sources into one batch
- **Batch Processing**: Prepare image lists for batch processing nodes
- **Workflow Integration**: Connect different image processing branches