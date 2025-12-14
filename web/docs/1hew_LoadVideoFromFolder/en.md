# Load Video From Folder

**Node Purpose:** Loads video files from a specified folder. Supports traversing subfolders and selecting videos by index.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `folder` | - | STRING | "" | - | The absolute path to the folder containing video files. |
| `index` | - | INT | 0 | -8192~8192 | The index of the video to load. Supports negative indexing (e.g., -1 for the last file). |
| `include_subfolder` | - | BOOLEAN | True | - | Whether to include video files in subfolders. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `VIDEO` | VIDEO | The loaded video object. |

## Features

- **Folder Traversal**: Can load videos from a flat folder or recursively from subfolders.
- **Indexing**: Select specific videos using an integer index. If the index exceeds the count, it wraps around (modulo operation).
- **Supported Formats**: Supports common video formats like `.webm`, `.mp4`, `.mkv`, `.mov`, `.avi`.
