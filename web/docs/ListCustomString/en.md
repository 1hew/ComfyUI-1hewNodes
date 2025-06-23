# List Custom String

**Node Function:** The `List Custom String` node generates string type lists, supporting dash separators and multiple delimiters for flexible text-to-string list conversion.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `custom_text` | - | STRING | "" | Multiline text | Custom text input with various separators |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string_list` | STRING | Generated string list |
| `count` | INT | Number of strings in the list |

## Function Description

### Separator Support
- **Dash separator priority**: When lines containing only dashes (--) are present, only dash separation is used, overriding other separators
- **Multiple delimiters**: Supports comma (,), semicolon (;), and newline (\n) separators
- **Chinese/English separators**: Supports both Chinese (，；) and English (,;) punctuation