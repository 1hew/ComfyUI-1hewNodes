# Image Batch Group - Windowing with overlap and fill

**Node Purpose:** `Image Batch Group` partitions an image batch into windows of size `batch_size` with configurable `overlap`. Controls how the last window is handled (drop, keep remaining, backtrack to end, or fill with color), and reports start indices, window counts, and valid counts.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `batch_size` | - | INT | 81 | 1-1024 | Window size per group. |
| `overlap` | - | INT | 0 | 0-1024 | Overlap frames between consecutive windows. |
| `last_batch_mode` | - | COMBO | `backtrack_last` | `drop_incomplete` / `keep_remaining` / `backtrack_last` / `fill_color` | Strategy for the final window. |
| `color` | - | STRING | `1.0` | grayscale/HEX/RGB/named | Background color when `fill_color` is used; supports `0.0-1.0`, `R,G,B`, HEX, and names (`red`, `white`, etc.). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | When `fill_color`, returns the padded batch; otherwise returns only original frames. |
| `group_total` | INT | Total number of groups. |
| `start_index` | LIST(INT) | Start indices for each window. |
| `batch_count` | LIST(INT) | Window sizes (after strategy application). |
| `valid_count` | LIST(INT) | Real frames per window (accounts for padding). |

## Features

- Parameter validation: prevents invalid overlaps; in `backtrack_last` mode allows `overlap <= batch_size`, otherwise requires `overlap < batch_size`.
- Start index computation: step size `batch_size - overlap`; guarded when step would be non-positive.
- Modes:
  - `drop_incomplete`: exclude incomplete tail windows.
  - `keep_remaining`: include the last partial window as-is.
  - `backtrack_last`: shift final start to align the last full window to the end.
  - `fill_color`: pad with solid frames so all windows are full-size; returns the padded batch.
- Color parsing: accepts grayscale floats, `R,G,B` (0–1 auto-converted), HEX, single-letter shortcuts (`r/g/b/c/m/y/k/w`), and named colors.
- Valid counts: last window reflects actual, non-padded frames when applicable.

## Typical Usage

- Sliding windows with overlap: set `batch_size=M`, `overlap=K` to build rolling windows for sequence processing.
- Keep only full windows: set `last_batch_mode=drop_incomplete`.
- Align final window to end: set `last_batch_mode=backtrack_last`.
- Pad to uniform window size: set `last_batch_mode=fill_color` and specify `color`.

## Notes & Tips

- When `fill_color`, alpha channel (if present) is set to `1.0`; grayscale padding uses the average of RGB.
- `valid_count` helps distinguish real versus padded frames, especially in the last window.