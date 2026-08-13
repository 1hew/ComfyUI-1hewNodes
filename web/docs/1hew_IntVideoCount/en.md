# Int Video Count - Count connected videos

**Node Purpose:** `Int Video Count` counts connected video objects from dynamic `video_1..video_N` inputs. It is useful for converting the number of optional video inputs into an integer parameter.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video_1` | - | VIDEO | - | - | First video input; the frontend dynamically appends `video_2..video_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of connected videos. |

## Features

- Dynamic inputs: connecting the last `video_X` port automatically appends the next video port.
- Empty inputs: disconnected video inputs are not counted.
- Ordered collection: dynamic video inputs are processed by their numeric suffix.

## Typical Usage

- Optional video count: output the number of connected videos for downstream integer controls.
- Dynamic workflows: use the count to control video batch or reference-video processing.

## Notes & Tips

- Each non-`None` video input counts as one connected video.
- The node counts connected video objects and does not inspect their frame content.
