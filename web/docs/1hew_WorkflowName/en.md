# Workflow Name - Derive workflow filename

**Node Purpose:** `Workflow Name` reads the current workflow path recorded by the workflow monitor and returns a processed name string. It supports prefix/suffix, date-based folder formatting, full-path output, and optional extension stripping.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `prefix` | - | STRING | - | - | Prefix appended to the workflow name. |
| `suffix` | - | STRING | - | - | Suffix appended to the workflow name. |
| `date_format` | - | COMBO | `yyyy-MM-dd` | `none` / `yyyy-MM-dd` / `yyyy/MM/dd` / `yyyyMMdd` / `yyyy-MM-dd HH:mm` / `yyyy/MM/dd HH:mm` / `yyyy-MM-dd HH:mm:ss` / `MM-dd` / `MM/dd` / `MMdd` / `dd` / `HH:mm` / `HH:mm:ss` / `yyyy年MM月dd日` / `MM月dd日` / `yyyyMMdd_HHmm` / `yyyyMMdd_HHmmss` | Date folder format used to build the final string. |
| `full_path` | - | BOOLEAN | False | - | When True, includes the directory part of the workflow path (normalized with `/`). |
| `strip_extension` | - | BOOLEAN | True | - | When True, removes the `.json` extension from the returned name. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Processed workflow name string. |

## Features

- Monitor integration: reads `current_workflow.tmp` produced by the workflow monitor utility.
- Name processing: extracts base filename (or full path), applies prefix/suffix, and optionally strips `.json`.
- Date folder building: when `date_format` is enabled, returns `date_str/original_name/result`.
- Robust file reading: retries several times to handle concurrent file writes.

## Typical Usage

- Use as `filename_prefix` for save nodes: combine with image/video saving to organize outputs by workflow and date.
- Produce stable naming across sessions: set `strip_extension=True` and select a deterministic `date_format`.

## Notes & Tips

- The workflow monitor writes `current_workflow.tmp`; keep the monitor running for continuous updates.
- `full_path=True` returns forward-slash normalized paths for cross-platform consistency.
