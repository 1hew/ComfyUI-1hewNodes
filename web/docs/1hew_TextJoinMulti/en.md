# Text Join Multi

**Node Function:** The `Text Join Multi` node is used to concatenate multiple text inputs into a single string, supporting custom separators and dynamic input referencing with intelligent comment filtering capabilities, commonly used for text merging and formatting processing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `text1` | Required | STRING | "" | Multi-line text | First text input, supports multi-line text and comments |
| `text2` | Required | STRING | "" | Multi-line text | Second text input, supports multi-line text and comments |
| `text3` | Required | STRING | "" | Multi-line text | Third text input, supports multi-line text and comments |
| `text4` | Required | STRING | "" | Multi-line text | Fourth text input, supports multi-line text and comments |
| `text5` | Required | STRING | "" | Multi-line text | Fifth text input, supports multi-line text and comments |
| `separator` | - | STRING | "\\n" | - | Text concatenation separator, defaults to newline |
| `input` | Optional | STRING | "" | - | Dynamic input value, can be referenced in text using {input} |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Concatenated text string with comments automatically filtered |

## Features

### Comment Processing
- **Line-start Comments**: Entire lines starting with `#` are completely removed
- **Inline Comments**: Content after `#` within a line is removed, preserving valid content before `#`
- **Multi-line Comments**: `"""..."""` and `'''...'''` comments are completely removed without generating excess blank lines
- **Blank Line Preservation**: Preserves original text blank line structure, only removing excess blank lines caused by comments