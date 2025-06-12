# Image Tile Merge

**Node Function:** The `Image Tile Merge` node is used to intelligently merge multiple image tiles back into a complete image, supporting multiple blend modes and overlap area processing, ensuring seamless stitching effects in merged images.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `tiles` | Required | IMAGE | - | - | Image tile batch to be merged |
| `tile_meta` | Required | DICT | - | - | Tile metadata from ImageTileSplit node |
| `blend_strength` | - | FLOAT | 0.5 | 0.0-1.0 | Blend strength controlling overlap area blending degree |
| `blend_mode` | - | COMBO[STRING] | linear | none, linear, gaussian | Blend mode: none (no blending), linear, gaussian |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `merged_image` | IMAGE | Complete merged image |

## Function Description

### Application Scenarios
- **AI processing result merging**: Merge AI-processed image tiles back into complete image
- **Super-resolution reconstruction**: Merge super-resolution processed image tiles
- **Large image processing**: Reconstruct large-sized images after tile processing
- **Batch image stitching**: High-quality image stitching and panoramic composition