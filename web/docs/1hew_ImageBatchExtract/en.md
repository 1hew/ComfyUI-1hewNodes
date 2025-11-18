# Image Batch Extract - Select frames by index/step/uniform

**Node Purpose:** `Image Batch Extract` selects specific frames from an input image batch using three modes: explicit index list, fixed step, or uniform sampling across the batch. Supports negative indices, large-batch chunked selection, and optional capping via `max_keep`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mode` | - | COMBO | `step` | `index` / `step` / `uniform` | Selection mode. |
| `index` | - | STRING | `0` | comma list | For `index` mode; comma-separated integers, supports negative indices (e.g., `-1`). |
| `step` | - | INT | 4 | 1-8192 | For `step` mode; take every `step` frames. |
| `uniform` | - | INT | 4 | 0-8192 | For `uniform` mode; select `uniform` frames evenly across the batch; `0` selects none. |
| `max_keep` | - | INT | 10 | 0-8192 | Cap number of selected frames; `0` keeps all. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch containing only the selected frames. |

## Features

- Index parsing: accepts English/Chinese commas; negative indices resolve relative to batch size.
- Step selection: pick frames at fixed stride `step` starting from `0`.
- Uniform sampling: evenly distribute `uniform` picks between `0` and `batch_size-1`.
- Robust filtering: invalid indices are skipped; returns empty batch when no valid indices remain.
- Chunked gather: selects in chunks (size `512`) to reduce memory spikes on large lists.

## Typical Usage

- Extract keyframes every N frames: set `mode=step`, `step=N`.
- Uniform thumbnails: set `mode=uniform`, `uniform=K` to sample K frames across the batch.
- Custom list: set `mode=index`, `index=0, 2, 10, -1` to include specific frames including the last.

## Notes & Tips

- `max_keep=0` keeps all selected frames; set `>0` to cap the output size.
- If all indices are out of range, the output is an empty batch with matching dtype/device.