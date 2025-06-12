# Prompt Extract

**Node Function:** The `Prompt Extract` node is used to extract content in specified languages from text, supporting JSON format and key-value pair format text parsing, commonly used for separation and extraction of multilingual prompts.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `text` | Required | STRING | "" | Multi-line text | Text containing multilingual content, supports JSON format or key-value pair format |
| `language` | - | COMBO[STRING] | en | en, zh | Language type to extract: en (English), zh (Chinese) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `text` | STRING | Extracted text content in specified language |

## Function Description

### Usage Examples

**Input Text Example 1 (JSON format):**
```json
{
  "English": "A beautiful landscape with mountains",
  "中文": "美丽的山景风光"
}