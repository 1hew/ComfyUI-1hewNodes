# List Text - Prefix and suffix for each list item

**Node purpose:** `List Text` adds a prefix and suffix to every item of a list and returns a list. It takes the same `prefix` / `suffix` inputs as `Text List to String`; the difference is the output. `Text List to String` joins everything into one string, while `List Text` keeps one string per item so the result can feed the next list node.

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `text_list` | optional | * | - | - | Items to decorate; a list, a single value, or nested lists (flattened one level) |
| `prefix` | - | STRING | `` | - | Text placed before each item |
| `suffix` | - | STRING | `` | - | Text placed after each item |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `text_list` | STRING list | One decorated string per item, in order |

## Example

| `text_list` | `prefix` | `suffix` | Output |
| ----------- | -------- | -------- | ------ |
| `["cat", "dog"]` | `a ` | ` !` | `["a cat !", "a dog !"]` |
| `[["x", "y"], "z"]` | `` | `` | `["x", "y", "z"]` |
| `[1, 2]` | `#` | `` | `["#1", "#2"]` |

## Notes

- Non-string items are converted with `str()` before the prefix and suffix are added.
- With no items connected the output is an empty list.
- Prefix and suffix are literal text: escape sequences are not processed here, the same as `Text List to String`.
