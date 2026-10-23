# Save Audio - Save audio to disk with optional metadata

**Node Purpose:** `Save Audio` saves an AUDIO input into the ComfyUI output or temp directory and returns the saved absolute file path. It supports FLAC / WAV / MP3 / Opus. The `audio` input is optional: when it is unconnected (or empty), the node finishes with an empty output and writes no file.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `audio` | optional | AUDIO | - | - | Audio to save (`waveform` shaped `[batch, channels, samples]` plus `sample_rate`); when absent, the node finishes with an empty output. |
| `filename` | - | STRING | `audio/ComfyUI` | - | Save prefix passed to ComfyUI path generation; typically supports date placeholders (e.g. `%date:yyyy-MM-dd%`). |
| `format` | - | COMBO | `mp3` | `flac` / `wav` / `mp3` / `opus` | Save format; defaults to mp3 (320 kbps, lossy) — choose flac / wav for lossless archiving; mp3 / opus are resampled when the encoder does not support the source rate. |
| `auto_increment` | - | BOOLEAN | `true` | - | When `true`, creates a new numbered file; when `false`, uses a fixed filename and overwrites it. |
| `save_output` | - | BOOLEAN | `true` | - | Save to output directory when enabled; save to the temp directory when disabled. |
| `save_metadata` | - | BOOLEAN | `true` | - | Write prompt/workflow metadata into the audio container when enabled. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `file_path` | STRING | Absolute file path of the saved audio; batch results are newline-separated. |

## Features

- Empty-input pass-through: an unconnected `audio` port, a `None` value, or an empty `waveform` / `sample_rate` returns an empty output without creating files, and sends an empty audio list so the frontend clears any player left over from the previous run.
- Batch splitting: the `waveform` batch dimension is written one file per item; in fixed-name mode the second file onward gets `_001`, `_002` suffixes.
- Channels: mono and stereo are supported; more channels raise a clear error instead of writing a broken file.
- Sample rates: flac / wav keep the source rate; mp3 / opus snap to the nearest encoder-supported rate and resample.
- Metadata: optionally writes prompt/workflow into the container metadata; skipped automatically when ComfyUI runs with `--disable-metadata`.
- In-node player: a frontend extension embeds an audio player in the node and points it at the freshly saved file (`/view?filename=...&type=output`) after each run, so results can be auditioned immediately. The player keeps its own DOM container, so it stays full-width on both the classic canvas and Nodes 2.0, and it hides itself when no audio was produced.
- UI payload: also returns `SavedAudios` (falling back to `PreviewAudio` on older builds) so the frontend can read the filename, subfolder, and folder type.

## Typical Usage

- Archive or deliver results from audio generators (TTS, music, video soundtracks).
- Disable `save_output` for audition-only runs so files land in the temp directory.
- Save a whole audio batch as one file per item, matching the input batch.

## Notes & Tips

- The default mp3 output is lossy at 320 kbps, which suits direct delivery and auditioning; if the saved file will be transcoded, mixed, or aligned downstream, switch to flac / wav to keep a lossless source.
- Encoding uses `PyAV` (bundled with ComfyUI); if a codec is missing from the build (e.g. `libopus`), that format fails — use flac or wav instead.
- `auto_increment=false` overwrites an existing file; double-check the target directory when saving batches.
- An absolute path prefix bypasses the output directory; when the saved file is outside ComfyUI's served directories the node falls back to `PreviewAudio(audio)`, which plays the input audio rather than the saved file (older builds may show no preview).
