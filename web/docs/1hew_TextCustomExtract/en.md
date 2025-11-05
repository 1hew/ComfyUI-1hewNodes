# Text Custom Extract

**Node Function:** The `Text Custom Extract` node extracts values of specified keys from JSON objects or arrays, supporting both precise matching and enhanced matching modes, with label filtering capabilities, ideal for data extraction and processing scenarios.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `json_data` | - | STRING | "" | Multiline text | JSON object or array data |
| `key` | - | STRING | "zh" | Text | Key name to extract |
| `precision_match` | - | COMBO[STRING] | disabled | disabled, enabled | Precision matching mode switch |
| `label_filter` | Optional | STRING | "" | Text | Filter by label values (comma separated, supports partial match) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Extracted value string |

## Function Description

### Label Filtering
- **Filter support**: Filter objects by label values before extracting specified keys
- **Multiple filters**: Support comma-separated multiple filter conditions
- **Partial matching**: Support partial string matching for flexible filtering
- **Chinese/English separators**: Support both Chinese (，) and English (,) comma separators

### Matching Modes
- **Precise matching (enabled)**: Only matches exactly identical key names
- **Enhanced matching (disabled)**: Supports multiple key name variants and language matching

### Enhanced Matching Features
- **Case variants**: Automatically matches different case combinations
- **Language mapping**: Intelligently recognizes Chinese-English key name correspondences
  - English related: en, English, eng, 英文, 英语
  - Chinese related: zh, Chinese, chn, 中文
- **Common patterns**: Supports common key name variants for coordinates, dimensions, etc.
  - Coordinates: x/X, y/Y, pos_x, position_x
  - Dimensions: width/w, height/h
  - Identifiers: id/ID, name/NAME