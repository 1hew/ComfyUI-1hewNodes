# List Custom Float

**Node Function:** The `List Custom Float` node generates floating-point type lists, supporting dash separators and multiple delimiters for flexible text-to-float list conversion.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `custom_text` | - | STRING | "" | Multiline text | Custom text input with various separators |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `float_list` | FLOAT | Generated floating-point list |
| `count` | INT | Number of floats in the list |

## Function Description

### Separator Support
- **Dash separator priority**: When lines containing only dashes (--) are present, only dash separation is used, overriding other separators
- **Multiple delimiters**: Supports comma (,), semicolon (;), and newline (\n) separators
- **Chinese/English separators**: Supports both Chinese (，；) and English (,;) punctuation