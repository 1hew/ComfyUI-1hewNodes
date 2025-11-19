# Image PingPong - Bidirectional Frame Repeat

**Node Purpose:** `Image PingPong` generates a back-and-forth sequence across a batch of frames: forward, backward, forward... with optional pre-reverse and link-frame removal for smoother transitions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input frame batch `(B, H, W, C)`; operates over the batch axis. |
| `pre_reverse` | - | BOOLEAN | `false` | - | Reverse the input batch before generating sequences. |
| `ops_count` | - | INT | `1` | `0-100000` | Number of segments to generate; e.g., `2 → forward, backward`; `3 → forward, backward, forward`. `0` means infinite with `frame_count` truncation. |
| `frame_count` | - | INT | `0` | `0-1000000` | Output truncation length; `0` disables truncation unless `ops_count=0`. |
| `remove_link_frame` | - | BOOLEAN | `true` | - | Remove the first frame of each subsequent segment to avoid duplicate junction frames. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Generated ping-pong sequence over the input batch. |

## Features

- Segment alternation: forward `[0..B-1]`, backward `[B-1..0]`, alternating per segment index.
- Smooth junctions: when enabled, drops the first frame of each segment after the first to avoid boundary duplicates.
- Pre-reverse option: reverses the input batch before sequence generation.
- Flexible limits: finite by `ops_count`, infinite when `ops_count=0` with `frame_count` truncation.

## Edge Cases

- `ops_count=0` and `frame_count=0` → empty output.
- `ops_count>0` and `frame_count=0` → exact concatenation of `ops_count` segments (with optional link removal).
- `ops_count=0` and `frame_count>0` → infinite alternating cycle truncated to `frame_count` frames.
- Both nonzero → generate then truncate; if generation is shorter than `frame_count`, use generated length.

## Typical Usage

- Create seamless loop segments for motion sequences by enabling `remove_link_frame`.
- Start from a backward-first aesthetic by enabling `pre_reverse`.
- Use `frame_count` to align clip length to downstream requirements.

## Notes & Tips

- Operates over batch frames only; pixel data is not modified beyond reindexing.
- For `B=1`, link-frame removal keeps single-frame stability (no duplicates).