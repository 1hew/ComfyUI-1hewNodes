# Workflow Name

**Node Function:** The `Workflow Name` node automatically retrieves the current workflow filename by monitoring temporary files. It supports path control, custom prefixes/suffixes, and date formatting, commonly used for dynamic workflow naming and file organization.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `prefix` | Optional | STRING | "" | - | Custom prefix to add before the workflow name |
| `suffix` | Optional | STRING | "" | - | Custom suffix to add after the workflow name |
| `date_format` | Optional | COMBO[STRING] | "yyyy-MM-dd" | Multiple formats | Date format for prefix: none, yyyy-MM-dd, yyyy/MM/dd, yyyyMMdd, yyyy-MM-dd HH:mm, yyyy/MM/dd HH:mm, yyyy-MM-dd HH:mm:ss, MM-dd, MM/dd, MMdd, dd, HH:mm, HH:mm:ss, yyyy年MM月dd日, MM月dd日, yyyyMMdd_HHmm, yyyyMMdd_HHmmss |
| `full_path` | Optional | BOOLEAN | True | True/False | Whether to include the full path (relative to workflows directory) or just the filename |
| `strip_extension` | Optional | BOOLEAN | True | True/False | Whether to remove the .json extension from the filename |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `string` | STRING | Processed workflow name with applied formatting options |
