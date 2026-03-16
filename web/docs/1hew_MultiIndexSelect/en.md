# Multi Index Select - Per-Column Index Picker

**Node Purpose:** `Multi Index Select` accepts dynamic `input_1..input_N` ports, and applies one shared `index` to select the corresponding item from each input column. It supports list-style values (e.g. text lists) and tensor batches (e.g. image/mask batches).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `index` | - | INT | 0 | -999999..999999 | Shared index used by all input columns. Supports negative index. |
| `input_1` | - | ANY (`*`) | - | - | First input column. |
| `input_2…input_N` | optional (dynamic, up to 10) | ANY (`*`) | - | - | Additional input columns, matched to `output_2…output_N`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output_1…output_N` | ANY (`*`) | Per-column selected value for the same `index`; input/output columns are matched by suffix number. |

## Features

- Column-wise mapping: `input_k` always maps to `output_k`.
- Shared index: one `index` controls selection for all connected columns.
- Mixed data support:
- Tensor batch input: selects by batch dimension and keeps batch shape as `1` item.
- List/tuple input: selects one element by index.
- Scalar/other input: passes through unchanged.
- Safe bounds behavior: index is clamped to valid range; negative index works as Python-style from end.

## Typical Usage

- Keep image and text synchronized: select the same row from `image_list` and `string_list`.
- Mix input forms: use `image_list` / `mask_list` together with image/mask batch inputs in one node.
- Multi-column selection: one node replaces multiple separate index-select steps.

## Notes & Tips

- This node is intended for list/batch style data. For scalar inputs, output equals input.
- Type mismatches are expected if connecting string outputs to image preview ports; connect each output to a compatible downstream node.
