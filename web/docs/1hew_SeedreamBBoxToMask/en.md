# Seedream BBox to Mask - Turn Seedream bboxes into masks

**Node Purpose:** `Seedream BBox to Mask` turns Seedream bboxes into rectangular masks. It handles
bboxes only: the `<bbox>x1 y1 x2 y2</bbox>` marker from a prompt and the `bounding_box` returned by
layer decomposition. `<point>x y</point>` markers belong to a different node and are ignored.

Only **two units** exist and `coord_mode` states which one is in use; nothing is auto-detected:

- `normalized` (default): per-mille 0-1000, forward `x = round(x_px / width * 1000)`, inverse
  `x_px = round(x / 1000 * width)`. Matches the prompt marker and `bounding_box.normalized`.
- `absolute`: pixels. Matches `bounding_box.absolute`.

Both use `[left, top, right, bottom]` (xyxy) where `right` / `bottom` are edges rather than
width/height, so there is no box_format option.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Reference image; only its width/height are used. All frames in a batch must share one size. |
| `coordinates_string` | - | STRING | empty | multiline | One entry per line, always `xyxy`; accepts `<bbox>` tags, `bbox(...)` calls, JSON arrays/objects, labeled lines (`xyxy: 1 2 3 4`) and bare numbers; a line starting with `#`, `//` or `--` is a whole-line comment. |
| `coord_mode` | - | COMBO | `normalized` | `normalized` / `absolute` | Coordinate unit: per-mille 0-1000 or pixels. |
| `output_mode` | - | COMBO | `separate` | `separate` / `merge` | `separate`: one mask per box (batch); `merge`: union of everything. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | `separate`: `N×H×W` with N parsed boxes; `merge`: `1×H×W`. With no boxes, `separate` returns `0×H×W` and `merge` an all-zero frame. |
| `count` | INT | Number of parsed boxes. |

## Features

- Inverts the documented mapping exactly: `px = round(n / 1000 * extent)`, clamped to the image.
- Parse order: tags, then open tags only, then `bbox(...)` calls, then JSON arrays/objects, then bare numbers per line.
- With tags present only the tagged numbers are read, so a full prompt line such as
  `2 图1<bbox>120 180 640 760</bbox> 区域替换成花园` yields exactly one box; line numbers and
  `图1` / `Image 1` prefixes never leak into the coordinate list.
- Layer decomposition: when `bounding_box` carries both `absolute` and `normalized` the field selected by
  `coord_mode` is used; when only one is present its own unit is honoured, so either JSON can be pasted as-is.
- In `normalized` mode a value above 1000 raises a clear error asking for `absolute` instead of silently
  producing a wrong mask.
- Comments are for humans: whole lines starting with `#`, `//` or `--` are dropped, so `<bbox>` markers or numbers inside them are never parsed.
- Do not encode the unit in a comment: `coord_mode` is the only source of truth, and a wrong comment changes nothing (and raises nothing).
- Full-width digits and punctuation are normalized to ASCII first.

## Typical Usage

- Official example: `<bbox>120 180 640 760</bbox>` on a 640×480 image becomes `[77, 86, 410, 365]`.
- Several boxes, one per line with `separate`, produce a mask batch for per-region processing.
- Self-documenting blocks are fine:

  ```text
  # Seedream bbox, xyxy, normalized 0-1000
  120 180 640 760
  # second box (switch coord_mode to absolute for pixels)
  500 400 999 999
  ```
- Overlay the `merge` mask on the source image to visually confirm a selection.
- Layer restore: paste the whole decomposition JSON; `absolute` has no rounding error and `normalized`
  fits any canvas (the API documents a ±1px rounding error).

## Notes & Tips

- Bboxes only: `<point>` markers and lines holding just two numbers are ignored (`count` stays unchanged).
- For layer decomposition, connect the **composited base image** size (e.g. 2048×2048), not the layer PNG's own
  `size` (e.g. 1273×265); `bounding_box` is always relative to the base image.
- The Seedream 5.0 pro editing API takes only `image` and `prompt`, with no mask parameter; the masks
  produced here are for local compositing, previewing or other models, not for upload.
- All frames in a batch must share one size; the masks use the first frame's dimensions.
