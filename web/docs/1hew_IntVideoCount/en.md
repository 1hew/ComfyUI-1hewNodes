# Int Video Count - Count non-empty videos

**Node Purpose:** `Int Video Count` counts non-empty video objects from dynamic `video_1..video_N` inputs. Emptiness follows `Any Empty Bool`, so `None` and empty values are ignored. It is useful for converting the number of optional video inputs into an integer parameter.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video_1` | optional | VIDEO | - | - | First video input; the frontend dynamically appends `video_2..video_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of non-empty videos. |

## Features

- Dynamic inputs: connecting the last `video_X` port automatically appends the next video port.
- Emptiness: `None` and empty values are ignored, matching `Any Empty Bool`.
- Ordered collection: dynamic video inputs are processed by their numeric suffix.
- List-aware: list outputs (OUTPUT_IS_LIST) are flattened and each non-empty item counts once.

## Typical Usage

- Optional video count: output the number of non-empty videos for downstream integer controls.
- Dynamic workflows: use the count to control video batch or reference-video processing.

## Notes & Tips

- Each non-empty video input counts as one; empty dicts or containers are ignored.
- The node evaluates presence only and does not inspect frame content.
