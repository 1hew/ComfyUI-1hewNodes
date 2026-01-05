# Get File Count - Count files in a folder

**Node Purpose:** `Get File Count` scans a folder and returns the number of image or video files found. It supports recursive scanning, stable sorting, and a change hash suitable for caching.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `folder` | - | STRING | `""` | - | Target folder path. |
| `type` | - | COMBO | `image` | `image` / `video` | File type group to count. |
| `include_subdir` | - | BOOLEAN | `true` | - | Include subfolders in the scan. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `count` | INT | Number of matched files. |
| `folder` | STRING | Echo of the input folder. |
| `include_subdir` | BOOLEAN | Echo of the input option. |

## Features

- Extension filtering: counts common image or video extensions.
- Deterministic order: paths are sorted case-insensitively for consistent results.
- Change tracking: a path-list hash is used to support efficient downstream caching.

## Typical Usage

- Count frames as files before batch loading images or videos from a directory.

## Notes & Tips

- When `folder` points to an empty or unmatched directory, `count` becomes `0`.

