# Image Add Label - Adaptive text labeling

**Node Purpose:** `Image Add Label` adds text labels to images with auto-scaling based on image size and placement direction. Supports batch images and batch texts, dynamic variable substitution, dashed section separators, and multi-line wrapping that fits available space.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `height_pad` | - | INT | 24 | 1–1024 | Minimum padding in pixels around text. Scales with image size. |
| `font_size` | - | INT | 36 | 1–256 | Base font size at 1024 reference resolution. Scales automatically. |
| `invert_color` | - | BOOLEAN | True | - | White label with black text when True; inverted when False. |
| `font` | - | COMBO | auto | fonts dir | Font file from `fonts/` directory (e.g., `Alibaba-PuHuiTi-Regular.otf`). |
| `text` | - | STRING(multiline) | "" | - | Label text. Supports `--` separator lines to split sections, otherwise uses newline. |
| `direction` | - | COMBO | `top` | `top`/`bottom`/`left`/`right` | Placement side of the label. |
| `input1` | - | STRING | "" | - | Optional variable available to template in `text`. |
| `input2` | - | STRING | "" | - | Optional variable available to template in `text`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Labeled image batch (`B×H×W×3`). |

## Features

- Auto scale: selects scale mode by content side and aspect, keeping perceived label proportion across sizes.
- Smart wrap: width-constrained multi-line wrapping for both space-separated and CJK text; caches measurements for performance.
- Fixed line height: consistent line spacing from font metrics, avoiding layout jitter across lines.
- Variable templates: supports `{input1}`, `{input2}`, and per-frame expressions `{index}`, `{idx}`, `{range±K}`, with zero-padding for `{range}` depending on batch length.
- Direction-aware layout: top/bottom create horizontal label bars; left/right rotate a temporary label and attach vertically.
- Async rendering: per-frame rendering runs in worker threads to keep UI responsive.

## Typical Usage

- Add a top banner: set `direction=top`, write multi-line text, and tune `height_pad` for spacing.
- Side annotation: set `direction=left/right` to attach a vertical label bar; content wraps by available height.
- Batch templating: use `text` like `Frame {index+1}\nID {range}` to auto-number each frame.
- Custom fonts: drop `.ttf/.otf` into `fonts/` and select via `font`.

## Notes & Tips

- Base sizing uses 1024 reference resolution; actual `font_size` and `height_pad` are scaled by image dimensions and placement side.
- `--` separator lines split sections strictly; when present, only dashed blocks are considered, inner newlines are preserved.
- Colors: `invert_color=True` creates white label with black text, `False` creates black label with white text.