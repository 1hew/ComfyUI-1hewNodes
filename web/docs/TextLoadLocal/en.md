# Text Load Local

**Node Function:** The `Text Load Local` node is used to load Python format prompt files from the prompt directory and its subdirectories, supporting bilingual output in Chinese and English, by calling the get_prompt function in Python files to retrieve prompt content, commonly used for prompt management and multilingual text processing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `file` | Required | COMBO[STRING] | - | Python file list | Select Python file to load, automatically scans prompt directory and subdirectories |
| `user_prompt` | Required | STRING | "" | Multiline text | Additional user prompt to be combined with Python file content |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `en_string` | STRING | English prompt string |
| `zh_string` | STRING | Chinese prompt string |