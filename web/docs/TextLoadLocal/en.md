# Text Load Local

**Node Function:** The `Text Load Local` node is used to load JSON format prompt files from the prompt directory and its subdirectories, supporting bilingual output in Chinese and English, building complete prompts according to the original order of keys in JSON, commonly used for prompt management and multilingual text processing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `file` | Required | COMBO[STRING] | - | JSON file list | Select JSON file to load, automatically scans prompt directory and subdirectories |
| `user_prompt` | Required | STRING | "" | Multiline text | Additional user prompt to be combined with JSON content |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `en_string` | STRING | English prompt string |
| `zh_string` | STRING | Chinese prompt string |