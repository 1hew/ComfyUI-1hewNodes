# Audio Duration - Get Audio Length

**Node Purpose:** `Audio Duration` outputs the length of an audio clip in seconds by reading `sample_rate` and `waveform` from the upstream audio object.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `audio` | - | AUDIO | - | - | Audio object with fields `sample_rate` and `waveform` (`[batch, channels, samples]`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `second` | FLOAT | Audio duration in seconds (`samples / sample_rate`). |

## Features

- Robust handling: returns `0.0` when audio is missing or invalid.
- Batch-agnostic: uses the samples axis only; channel count and batch do not affect the result.
- Works with mono or stereo audio; no assumptions about sample dtype.

## Typical Usage

- Combine with video-saving nodes to annotate clip length.
- Validate audio preprocessing results by checking expected duration.

## Notes & Tips

- Ensure upstream audio provides `sample_rate > 0` and a `waveform` tensor.
- For streaming or segmented audio, pass the final merged audio to get the total length.