# Int Text Count - Count non-empty text-like inputs

**Node Purpose:** `Int Text Count` counts non-empty `STRING` / `INT` / `FLOAT` inputs from dynamic wildcard `text_1..text_N` ports. Emptiness follows `Any Empty Bool`: blank strings, `0` / `0.0`, and `False` are ignored; `True` counts as one. Other value types (images, masks, audio, dictionaries, …) are not counted.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text_1` | optional | ANY (`*`) | - | - | First text-like input; the frontend dynamically appends `text_2..text_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of non-empty text-like inputs. |

## Features

- Dynamic inputs: connecting the last `text_X` port automatically appends the next text port.
- Wildcard ports: connect STRING, INT, or FLOAT (BOOLEAN is treated as an integer) to the same socket family.
- Emptiness follows `Any Empty Bool`: `""`, whitespace-only strings, `0`, `0.0`, and `False` are empty.
- List-aware: a string list (OUTPUT_IS_LIST, e.g. from `List Custom String`) contributes one count per non-empty element.
- Single pass: list inputs are counted inside one execution instead of mapping the node per item.

## Typical Usage

- Optional text count: count how many reference texts or numeric parameters were actually supplied.
- Dynamic parameters: drive downstream integer controls from the number of non-empty text inputs.

## Notes & Tips

- Because `0` counts as empty, a numeric zero is treated as "not provided". Use `Any Empty Int` if a different mapping is needed.
- Non-text values are ignored rather than raising an error, so a stray IMAGE/MASK connection simply does not contribute.
- The node does not measure character length or line count; it counts inputs.
