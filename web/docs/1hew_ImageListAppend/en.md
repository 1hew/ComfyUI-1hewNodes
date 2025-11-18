# Image List Append - Concatenate image lists

**Node Purpose:** `Image List Append` merges multiple image inputs into a single list, preserving order by input suffix (`image_1`, `image_2`, ...). Accepts either single images or lists of images for each input.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_1` | optional | IMAGE/LIST | - | - | First image or image list. |
| `image_2` | optional | IMAGE/LIST | - | - | Second image or image list. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_list` | IMAGE_LIST | Combined list of images, ordered by input suffix. |

## Features

- Flexible inputs: each input may be a single image or a list; `None` inputs are ignored.
- Deterministic ordering: inputs are ordered by numeric suffix (`image_1`, `image_2`, ...).
- Pass-through lists: lists are concatenated without copying frames.

## Typical Usage

- Merge outputs of different extractors into one list for downstream processing.
- Build variable-length sequences by combining multiple branches.

## Notes & Tips

- Output is a Python-style list of images; connect to nodes that accept image lists.