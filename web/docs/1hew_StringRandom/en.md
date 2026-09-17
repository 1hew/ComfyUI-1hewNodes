# String Random - Random Pick from Text

**Node Purpose:** `String Random` splits `custom_text` into candidate items using common separators, then randomly picks one item using the provided `seed`. It outputs the chosen string, useful for randomized prompts or parameters.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `custom_text` | - | STRING | `` (empty) | - (multiline) | Text containing candidate items, split by dash lines, newlines, `。`/`.` , `；`/`;` , or `，`/`,` . |
| `seed` | - | INT | 42 | 0–1125899906842624 | RNG seed used for a reproducible random pick. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Randomly chosen item from the text. |

## Features

- Separator priority: a line of dashes (`---`) splits first, then newlines, then `。`/`.` , then `；`/`;` , and finally `，`/`,` .
- Quote trimming: surrounding matching single/double quotes are stripped from each item.
- Reproducible: the pick is deterministic given the same `seed`.
- Empty handling: blank input yields an empty string output.

## Typical Usage

- Randomize a prompt segment among several candidate phrases.
- Pick a random style keyword or parameter value in batch or iterative workflows.

## Notes & Tips

- Items are whitespace-trimmed and empty items are discarded before the random pick.
- The `seed` is also passed through to the output's UI text so the selection is visible.
