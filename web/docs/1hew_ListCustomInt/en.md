# List Custom Int

**Node Function:** The `List Custom Int` node generates integer type lists, supporting dash separators and multiple delimiters for flexible text-to-integer list conversion.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `custom_text` | - | STRING | "" | Multiline text | Custom text input with various separators |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `int_list` | INT | Generated integer list |
| `count` | INT | Number of integers in the list |

## Function Description

### Separator Support
- **Dash separator priority**: When lines containing only dashes (--) are present, only dash separation is used, overriding other separators
- **Multiple delimiters**: Supports comma (,), semicolon (;), and newline (\n) separators
- **Chinese/English separators**: Supports both Chinese (，；) and English (,;) punctuation