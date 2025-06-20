# Text Custom List

**Node Function:** The `Text Custom List` node generates custom text/numeric lists, supporting multiple separators and data type conversion, compatible with other nodes for batch processing.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `custom_text` | - | STRING | "" | Multiline text | Custom text content supporting multiple separators |
| `type` | - | COMBO[STRING] | string | int, float, string | Output data type |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `list` | * | Parsed data list |
| `count` | INT | Number of list items |

## Function Description

### Separator Support
- **Comma separation**: Use comma(,) to separate multiple items
- **Semicolon separation**: Use semicolon(;) to separate multiple items
- **Newline separation**: Use newline characters to separate multiple items
- **Visual separation**: Supports any length of hyphenated lines such as `-`, `--`, and `---` as visual separators, without affecting the final list content.
- **Mixed separation**: Support mixed use of multiple separators

### Quote Processing
- **Double quote wrapping**: Automatically recognize and remove double quote wrapped text
- **Single quote wrapping**: Automatically recognize and remove single quote wrapped text
- **Smart parsing**: Maintain integrity of text within quotes

### Data Type Conversion
- **String type**: Maintain original text format
- **Integer type**: Automatically convert to integer, support float to int conversion
- **Float type**: Convert to floating-point format
- **Error handling**: Use default values (0 or 0.0) when conversion fails