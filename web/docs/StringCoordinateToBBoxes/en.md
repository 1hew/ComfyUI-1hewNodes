# String Coordinate to BBoxes

**Node Function:** The `String Coordinate to BBoxes` node converts string format coordinate lists to BBOXES format, supporting multiple input formats like "x1,y1,x2,y2" or "[x1,y1,x2,y2]" or multi-line coordinates, commonly used for coordinate data conversion and SAM2 compatibility.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `coordinates_string` | Required | STRING | "" | Multi-line text | Coordinate string input, supports "[x1,y1,x2,y2]" or multi-line coordinates |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `bboxes` | BBOXES | Converted bounding boxes in SAM2 compatible format |