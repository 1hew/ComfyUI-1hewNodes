# List Custom Seed - Custom Seed List

**Node Purpose:** `List Custom Seed` generates a list of unique random seeds seeded by the input `seed`, and returns the list alongside its count. Seeds are clamped to the valid range.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `seed` | - | INT | `42` | `0–1125899906842624` | Initial RNG seed; used to seed the Python RNG for reproducibility. |
| `count` | - | INT | `3` | `1–1000` | Number of unique seeds to generate. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `seed_list` | LIST | Generated unique seeds (list output). |
| `count` | INT | Length of `seed_list`. |

## Features

- Reproducible generation: RNG is seeded with the provided `seed`.
- Uniqueness within a run: avoids duplicates using an internal set and a bounded retry strategy.
- Range safety: each seed is clamped to `0..1125899906842624`.
- Core logic: execution; unique generation; clamping.

## Typical Usage

- Generate `N` seeds: set `seed=12345` and `count=8` for eight reproducible seeds.
- Batch workflows: use the list output to drive per-item randomness.

## Notes & Tips

- The node enforces uniqueness for the requested `count` using a retry cap (`count×10`) and a fallback strategy.
- Each invocation is independent; provide a deterministic `seed` to reproduce outputs.