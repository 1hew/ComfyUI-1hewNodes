# Image BBox Overlay by Mask

**Node Function:** The `Image BBox Overlay by Mask` node generates detection boxes based on masks and overlays them on images as outlines, supporting both independent and merge modes, capable of generating separate bounding boxes for each independent mask region or merging all masks into one bounding box.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to overlay bounding boxes on |
| `mask` | Required | MASK | - | - | Mask used to generate bounding boxes |
| `bbox_color` | - | COMBO[STRING] | red | red, green, blue, yellow, cyan, magenta, white, black | Bounding box color |
| `line_width` | - | INT | 3 | 1-20 | Bounding box line width |
| `padding` | - | INT | 0 | 0-50 | Bounding box padding pixels |
| `output_mode` | - | COMBO[STRING] | separate | separate, merge | Output mode: separate (independent mode), merge (merge mode) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image with bounding boxes overlaid |