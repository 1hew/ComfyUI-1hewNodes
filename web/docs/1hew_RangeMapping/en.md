# Range Mapping

**Node Function:** The `Range Mapping` node maps slider values in the 0-1 range to specified numerical ranges, supporting precision control and type conversion.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `value` | - | FLOAT | 1.0 | 0.0-1.0 | Slider input value |
| `min` | - | FLOAT | 0.0 | Unlimited | Minimum value of mapping range |
| `max` | - | FLOAT | 1.0 | Unlimited | Maximum value of mapping range |
| `rounding` | - | INT | 3 | 0-10 | Decimal precision control |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `float` | FLOAT | Mapped floating-point value |
| `int` | INT | Mapped integer value |

## Function Description

### Value Mapping
- **Linear mapping**: Linearly maps 0-1 range to min-max range
- **Real-time calculation**: Updates mapped values in real-time as slider changes
- **Dual output**: Provides both floating-point and integer outputs