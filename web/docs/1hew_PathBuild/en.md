# Path Build

**Node Function:** The `Path Build` node provides hierarchical path selection dropdown, supporting combination of predefined paths and custom path extensions. Ideal for file path management and hierarchical data selection scenarios.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `preset_path` | - | COMBO[STRING] | Dynamically generated | Predefined path list | Hierarchical path selection dropdown |
| `additional_path` | - | STRING | "" | Custom text | Additional path extension, optional custom field |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `full_path` | STRING | Complete path string |