# Get File Count

**Node Purpose:** Counts the number of files of a specific type (Image or Video) in a folder.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `folder` | - | STRING | "" | - | The absolute path to the folder to check. |
| `type` | - | COMBO | "image" | "image" / "video" | The type of files to count. |
| `include_subfolder` | - | BOOLEAN | True | - | Whether to search in subfolders. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `count` | INT | The total count of matching files found. |

## Features

- **Format Filtering**: Counts files based on selected type:
  - **Image**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`
  - **Video**: `.webm`, `.mp4`, `.mkv`, `.gif`, `.mov`, `.avi`
- **Utility**: Useful for iterating through datasets or validating folder contents before processing.
