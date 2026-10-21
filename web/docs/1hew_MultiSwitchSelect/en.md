# Multi Switch Select - Multi-Output Routing Selector

**Node Purpose:** `Multi Switch Select` uses a 1-based `select` expression to choose one or more inputs from dynamic `input_1..input_N`, then emits each chosen value only on the matching `output_k`. All other outputs remain `None`.

**Shared rules:** `select` and `Image Batch Index`'s `index` use the same parser (`utils/_index_range.py`) and accept exactly the same range syntax. The differences are that `select` is 1-based and its results are deduplicated and sorted.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `select` | - | STRING | `1` | 1..10 index/range expressions | Same syntax as `Image Batch Index`'s `index`: `1,3`, `1-3`, `[1,3]`, `[1,3)`, `:step`, negative indices |
| `input_1` | optional | ANY (`*`) | - | - | First candidate input |
| `input_2…input_10` | optional (dynamic) | ANY (`*`) | - | - | Additional candidate inputs |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output_1…output_10` | ANY (`*`) | Only the `output_k` matching an index in `select` carries the value of `input_k`; all others are `None` |

## Supported Syntax (same as Image Batch Index)

| Syntax | Result |
|--------|--------|
| `1`, `1,3`, `1 3`, `1.3`, `1，3` | Single value or list; `.`, `、` and whitespace are skipped as separators |
| `1-3`, `1 - 3` | Inclusive range: 1, 2, 3 |
| `1-5:2` | Step 2: 1, 3, 5 |
| `[1,3]`, `【1，3】` | Inclusive range: 1, 2, 3 |
| `[1,3)` | Left-closed, right-open: 1, 2 |
| `(1,3]` | Left-open, right-closed: 2, 3 |
| `(1,3)` | Open range: 2 |
| `[1,5]:2` | Step 2: 1, 3, 5 |
| `[1:3]` | 1 and 3 (`:` is only step syntax; it does not expand 1, 2, 3) |
| `-1` | Last connected input; e.g. with `input_1..input_4` connected, this is `input_4` |
| `[8,12]` | Out-of-range parts are dropped; only 8, 9, 10 remain |

- `[1:3]` is not a range: `:` belongs to the step syntax; use `[1,3]` or `1-3` for a range.
- Only `-` separates a range; `1~3` yields 1 and 3, not a range.
- Negative indices resolve against the highest connected input slot: with 1..4 connected, `-1` = `input_4` and `-4` = `input_1`.
- Out-of-range indices are skipped; an empty or fully invalid `select` selects nothing (all `output` values are `None`).
- Results are deduplicated and sorted; duplicates and input order do not affect the final selection.

## Features

- Multi-selection: `select=1,2,4` evaluates `input_1`, `input_2` and `input_4`; `input_3` is never evaluated.
- Fixed routing: `input_k` always maps to `output_k`.
- Lazy evaluation: `check_lazy_status` requests every selected input that is connected but unresolved in one pass; unselected branches are never added to the execution graph.
- Missing-input safety: a selected but unconnected input is not requested, its output stays `None`, and no `NodeInputError` is raised.
- Shared grammar: both nodes use `utils/_index_range.py`, so their range syntax cannot drift apart.
- Backward compatible: a numeric `select` saved by older workflows (e.g. `2`) still works because ComfyUI converts integer widget values to strings.

## Typical Usage

- Release several branches at once: `select=1,2,4` runs branches 1/2/4 and skips branch 3.
- Contiguous range: `select=1-3` runs branches 1/2/3.
- Only 1 and 3: `select=1,3`, `select=[1:3]` or `select=[1,3]:2`.
- Last connected branch: `select=-1`.
- Dynamic routing driven by another logic node that produces the `select` string.

## Notes & Tips

- Unselected outputs are `None` (not empty tensors). Downstream nodes connected to those `output_j` ports still execute and receive `None`, which usually errors downstream; only connect the outputs covered by the current `select`.
- A selected but unconnected `input_k` does not trigger its upstream branch, and `output_k` is `None`.
- An empty or invalid `select` does not fall back to 1; it selects nothing. Keep the default `1` if you want the first branch selected.
- The front-end exposes as many output slots as dynamic inputs (1..10); using `output_k` requires the matching `input_k` slot to exist or be connected.
- Unselected branches are not executed, but their links still have to pass prompt validation (types and required inputs).
