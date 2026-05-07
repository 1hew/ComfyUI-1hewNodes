# Int Image Count - Count valid images

**Node Purpose:** `Int Image Count` counts valid images from dynamic `image_1..image_N` inputs. Empty inputs, empty tensors, and fully black / all-zero images are ignored. It is useful for turning optional reference images into an integer parameter.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_1` | - | IMAGE | - | - | First image input; the frontend dynamically appends `image_2..image_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of valid images. |

## Features

- Dynamic inputs: connecting the last `image_X` port automatically appends the next image port.
- Validity check: disconnected values, empty tensors, and all-zero / black images are ignored.
- Batch-compatible: batched image inputs are checked and counted frame by frame.

## Typical Usage

- Reference image count: output the actual number of valid optional reference images.
- Dynamic parameters: drive downstream integer controls from the number of usable images.

## Notes & Tips

- The validity logic matches the reference-image filtering semantics used by `Gemini 3.1 Flash Image Preview`.
- If a pure black image should be treated as valid content, add a tiny non-zero value first or use another explicit counting method.
