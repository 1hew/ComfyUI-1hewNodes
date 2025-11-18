# String Coordinate to BBoxes - Parse bbox strings to list

**Node Purpose:** `String Coordinate to BBoxes` parses a multiline string of bbox coordinates into a nested list structure suitable for downstream consumers.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `coordinates_string` | - | STRING | `` | multiline | Lines of bbox coordinates; separators can be spaces or commas; brackets `[]()` are ignored. Each line should contain at least four numbers `x1 y1 x2 y2`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bboxes` | BBOXES | Nested list of bboxes in the form `[ [ [x1,y1,x2,y2], ... ] ]`. Returns `[[]]` when none are parsed. |

## Features

- Robust parsing: strips brackets and tolerates commas/spaces; converts floats to ints; ignores invalid tokens.
- Structure compatibility: produces a nested list suitable for consumers expecting grouped bboxes.

## Typical Usage

- Provide one bbox per line (e.g., `12,34,200,180`). The output groups all lines into a single inner list.

## Notes & Tips

- When no valid boxes are found, the output is `[[]]`.