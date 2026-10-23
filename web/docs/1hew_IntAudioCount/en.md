# Int Audio Count - Count non-empty audio inputs

**Node Purpose:** `Int Audio Count` counts audio objects from dynamic `audio_1..audio_N` inputs. Emptiness follows `Any Empty Bool`: `None` and empty containers are ignored, and a non-empty AUDIO dictionary (the standard `{waveform, sample_rate}` pair) counts as one input.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `audio_1` | optional | AUDIO | - | - | First audio input; the frontend dynamically appends `audio_2..audio_N` as connections are made. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Number of non-empty audio inputs. |

## Features

- Dynamic inputs: connecting the last `audio_X` port automatically appends the next audio port.
- Emptiness follows `Any Empty Bool`'s dictionary rule: any non-empty AUDIO dict counts as one.
- List-aware: an audio list (OUTPUT_IS_LIST) contributes one count per non-empty element.
- Single pass: list inputs are counted inside one execution instead of mapping the node per item.

## Typical Usage

- Optional audio count: check how many audio tracks were supplied before mixing or saving.
- Dynamic parameters: drive downstream integer controls from the number of audio inputs.

## Notes & Tips

- Presence is checked, not signal content: a silent (all-zero) waveform still counts because its AUDIO dict is non-empty.
- The node does not decode audio or inspect `waveform`/`sample_rate`.
- The batch dimension inside one AUDIO waveform is not split; one AUDIO input counts once.
