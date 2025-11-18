# Workflow Name - Current Workflow Path/Name with Date

**Node Purpose:** `Workflow Name` reads the current workflow path from a monitored temp file and formats the output with optional directory prefix for date, plus configurable `prefix`/`suffix` and extension handling.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `prefix` | - | STRING | `` | - | Text prepended to the workflow name. |
| `suffix` | - | STRING | `` | - | Text appended to the workflow name. |
| `date_format` | - | COMBO | `yyyy-MM-dd` | many | Date folder prefix; set to `none` to disable. |
| `full_path` | - | BOOLEAN | `False` | - | Output full path (dir + name) or only file name. |
| `strip_extension` | - | BOOLEAN | `True` | - | Remove `.json` extension from the output name. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Formatted workflow path or name, optionally prefixed by date. |

## Features

- Monitored temp file: reads `current_workflow.tmp` from candidate locations under `custom_nodes`.
- Retry-on-lock: retries on `PermissionError` with small delays.
- Path processing: normalize slashes, choose full path or basename, handle `.json` extension.
- Naming controls: apply `prefix`/`suffix`, optional extension stripping.
- Date prefix: adds a date directory based on selected format.
- Dynamic fingerprint: ensures fresh evaluation via a time-based fingerprint.

## Typical Usage

- Full path with date: keep `full_path=True`, `strip_extension=True`, and `date_format=yyyy-MM-dd` to organize saved results under dated folders.
- Name only: set `full_path=False` to get just the file name with your `prefix`/`suffix`.
- Preserve extension: set `strip_extension=False` when downstream consumers expect `.json`.

## Notes & Tips

- Ensure the monitoring script writes to `current_workflow.tmp` so the node can read the latest workflow path.
- When `date_format=none`, the output omits the date folder while retaining other formatting choices.