# String Coordinate to BBoxes

**Node Function:** The `String Coordinate to BBoxes` node converts string format coordinate lists to BBOXES format, supporting multiple input formats including "[x1,y1,x2,y2]" or multi-line coordinates for SAM2 compatibility.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `coordinates_string` | Required | STRING | "" | Multi-line text | String format coordinates supporting "[x1,y1,x2,y2]" or multi-line coordinates |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `bboxes` | BBOXES | SAM2 compatible bounding boxes format |
