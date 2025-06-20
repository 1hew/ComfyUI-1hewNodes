# Text Custom Extract

**Node Function:** The `Text Custom Extract` node extracts values of specified keys from JSON objects or arrays, supporting both precise matching and enhanced matching modes, ideal for data extraction and processing scenarios.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `json_data` | - | STRING | "" | Multiline text | JSON object or array data |
| `key` | - | STRING | "zh" | Text | Key name to extract |
| `precision_match` | - | COMBO[STRING] | disabled | disabled, enabled | Precision matching mode switch |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `value` | * | Extracted value list |

## Function Description

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

### JSON Parsing Capabilities
- **Standard JSON**: Supports standard JSON format parsing
- **Key-value format**: Supports simplified key-value pair text
- **Array processing**: Automatically traverses each object in JSON arrays
- **Error-tolerant parsing**: Uses regular expressions for error-tolerant processing

### Use Cases
- **Configuration extraction**: Extract specific parameters from configuration files
- **Data parsing**: Process JSON data returned by APIs
- **Multilingual support**: Extract multilingual text content
- **Coordinate extraction**: Extract coordinate information from position data