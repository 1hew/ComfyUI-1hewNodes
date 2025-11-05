# String Coordinate to BBox Mask

**Node Function:** The `String Coordinate to BBox Mask` node converts string format coordinate lists to BBoxMask format, supporting multiple input formats and requiring image input to obtain width and height information for accurate mask generation.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to get dimensions from |
| `coordinates_string` | Required | STRING | "" | Multi-line text | Coordinate string in format "x1,y1,x2,y2" or "[x1,y1,x2,y2]", supports multi-line coordinates |
| `output_mode` | - | COMBO[STRING] | merge | separate, merge | Output mode: separate (each coordinate line as individual mask), merge (all coordinates combined into one mask) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `bbox_mask` | MASK | Generated bounding box mask based on coordinates |