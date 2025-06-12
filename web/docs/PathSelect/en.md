# Path Select

**Node Function:** The `Path Select` node provides hierarchical path selection dropdown, supporting combination of predefined paths and custom fields.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `path` | - | COMBO[STRING] | Dynamically generated | Predefined path list | Hierarchical path selection |
| `filename` | - | STRING | "" | Custom text | Fourth-level custom field |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `filename` | STRING | Complete path string |