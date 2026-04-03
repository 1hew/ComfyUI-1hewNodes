# Multi Switch Select - Multi-Output Routing Selector

**Node Purpose:** `Multi Switch Select` uses a 1-based `select` value to choose one input from dynamic `input_1..input_N`, then emits that value only on the matching `output_k`. All other outputs remain `None`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `select` | - | INT | 1 | 1-10 | Input number to route; `1` maps to `input_1`. |
| `input_1` | - | ANY (`*`) | - | - | First candidate input. |
| `input_2…input_10` | optional (dynamic) | ANY (`*`) | - | - | Additional candidate inputs. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output_1…output_10` | ANY (`*`) | Only the `output_k` matching `select` carries the selected value; all others are `None`. |

## Features

- 1-based selection: `select=1` targets `input_1`, `select=2` targets `input_2`.
- Fixed routing: `input_k` always maps to `output_k`.
- Lazy evaluation: only the selected input branch is requested; if that input is unconnected, the node does not keep waiting and the matching output stays `None`.
- Safe bounds: `select` is automatically clamped to the `1..10` range.

## Typical Usage

- Placeholder outputs for multiple branches while preserving per-branch output slots.
- Mutually exclusive branch routing where only one output should carry data.
- Dynamic resource switching driven by another logic node that produces `select`.

## Notes & Tips

- Connect downstream nodes to the specific `output_k` you want to observe.
- If the selected input is not connected, the matching output will be `None`.
