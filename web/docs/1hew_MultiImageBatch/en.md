# Multi Image Batch

**Node Function:** The `Multi Image Batch` node builds an image batch from dynamic `image_X` inputs, unifying sizes based on the first image and applying a selected fit mode (`pad`, `crop`, `stretch`). It supports edge-aware padding and consistent ordering.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image_1` | Optional | IMAGE | - | - | First dynamic image input; supports additional `image_2`, `image_3`, ... |
| `fit` | Required | COMBO[STRING] | pad | crop, pad, stretch | Size unification mode |
| `pad_color` | Required | STRING | 1.0 | Color string | Padding color; supports multiple formats, `edge`/`e` uses edge average color |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `images` | IMAGE | Unified image batch with values clamped to `[0, 1]` |
