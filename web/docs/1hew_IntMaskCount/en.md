# Int Mask Count - Count valid masks

**Node Purpose:** `Int Mask Count` counts valid masks from dynamic `mask_1..mask_N` inputs. Empty inputs, empty tensors, and fully black / all-zero masks are ignored, using the same emptiness rule as `Any Empty Bool`. It turns optional masks into an integer parameter.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask_1` | optional | MASK | - | - | First mask input; the frontend dynamically appends `mask_2..mask_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of valid masks. |

## Features

- Dynamic inputs: connecting the last `mask_X` port automatically appends the next mask port.
- Emptiness follows `Any Empty Bool`: `None`, empty tensors, and all-zero entries are ignored.
- Batch-aware: a `[B, H, W]` mask batch contributes one count per valid frame.
- List-aware: mask lists (OUTPUT_IS_LIST) are flattened; each element is checked frame by frame, so `Mask Batch to List` can feed this node directly.
- Single pass: list inputs are counted inside one execution instead of mapping the node per item.

## Typical Usage

- Reference mask count: know how many optional masks were actually provided.
- Dynamic parameters: drive downstream integer controls from the number of usable masks.

## Notes & Tips

- An intentionally all-zero mask is treated as empty, exactly like `Any Empty Bool`.
- Only tensor contents are inspected; no resizing, cropping, or decoding is performed.
