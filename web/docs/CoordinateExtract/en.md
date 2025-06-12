# Coordinate Extract

**Node Function:** The `Coordinate Extract` node is used to extract X and Y coordinate lists from JSON-formatted coordinate data, supporting batch coordinate processing.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `coordinates_json` | - | STRING | - | Multi-line text | JSON-formatted coordinate data |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `x` | FLOAT | X coordinate list |
| `y` | FLOAT | Y coordinate list |

## Function Description

### Input Example
```json
[
    {
        "x": 0,
        "y": 512
    },
    {
        "x": 59,
        "y": 510
    }
]