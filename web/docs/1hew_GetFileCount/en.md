# Get File Count - Count files in a folder

**Node Purpose:** `Get File Count` scans a folder and returns the number of image, video, PSD, or txt files found, plus list-form `filename` and `file_path` outputs. `filename` can be configured to keep or strip the extension. It supports recursive scanning, stable sorting, and a change hash suitable for caching.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `folder` | - | STRING | `""` | - | Target folder path. |
| `type` | - | COMBO | `image` | `image` / `video` / `psd` / `txt` | File type group to count. |
| `filename_suffix` | - | BOOLEAN | `false` | - | Controls whether `filename` keeps the extension; when disabled, only the stem is returned, and when enabled the suffix is preserved. |
| `include_subdir` | - | BOOLEAN | `true` | - | Include subfolders in the scan. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `count` | INT | Number of matched files. |
| `folder` | STRING | Echo of the input folder. |
| `filename` | STRING LIST | List of matched file names, one item per file. |
| `file_path` | STRING LIST | List of matched file paths, one item per file. |
| `include_subdir` | BOOLEAN | Echo of the input option. |

## Features

- Extension filtering: counts common image, video, PSD, or txt extensions.
- Deterministic order: paths are sorted case-insensitively for consistent results.
- Filename output: exposes the sorted match list as a `filename` string list, with configurable extension keep/strip behavior.
- Path output: exposes the sorted match list as a `file_path` string list for precise downstream file access.
- Change tracking: a path-list hash is used to support efficient downstream caching.

## Typical Usage

- Count frames as files before batch loading images or videos from a directory.
- Scan a PSD asset folder, then pass `filename` directly into downstream nodes that accept string lists without an extra `List Custom String` step.
- When you need clean asset names without suffixes, keep `filename_suffix=false` to get stem-only multiline output.
- When downstream logic needs exact file locations, use `file_path` instead of reconstructing paths from `folder` and `filename`.

## Notes & Tips

- When `folder` points to an empty or unmatched directory, `count` becomes `0`.
- `filename` returns base file names only; when recursive scanning is enabled, duplicate names from different subfolders will appear as repeated list items.
- When `filename_suffix=false`, only the last extension is removed, so `a.test.psd` becomes `a.test`.
- `file_path` returns the matched path list and preserves subfolder information during recursive scans.

