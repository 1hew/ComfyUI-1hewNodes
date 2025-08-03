# Image Get Size

**Node Function:** The `Image Get Size` node is used to extract the width and height dimensions from an input image. This utility node is essential for workflows that need to know image dimensions for further processing, calculations, or conditional operations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image tensor from which to extract dimensions |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `width` | INT | The width of the image in pixels |
| `height` | INT | The height of the image in pixels |

## Features

### Automatic Processing
- **Dimension Handling**: Automatically handles both 3D and 4D image tensors
- **Batch Processing**: Works with batched images, extracting dimensions from the first image in the batch
- **Integer Output**: Returns precise integer values for width and height
- **Error-Free Processing**: Robust handling of different image tensor formats

### Technical Details
- Supports image tensors in format (batch, height, width, channels)
- Automatically adds batch dimension if input is 3D
- Returns dimensions as integer values for compatibility with other nodes
- Processes the first image in batch for dimension extraction

### Application Scenarios
- **Image Analysis**: Get image dimensions for analysis workflows
- **Conditional Processing**: Branch workflows based on image size
- **Dynamic Scaling**: Calculate scaling factors based on original dimensions
- **Layout Planning**: Plan image layouts based on dimensions
- **Quality Control**: Verify image dimensions meet requirements