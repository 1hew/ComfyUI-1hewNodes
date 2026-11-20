# List Math Expression - Element-wise math on lists

**Node purpose:** `List Math Expression` is the core `Math Expression` applied to lists. The expression grammar, the function set and the result checks are the same; the difference is that every input is a list, the expression runs once per position, and the outputs are lists. It pairs with `Text Path Extract` (list `value`) and `Text Path Replace` (list `replace`).

## Inputs

| Parameter | Socket | Type | Default | Range | Description |
| --------- | ------ | ---- | ------- | ----- | ----------- |
| `expression` | - | STRING | `a + b` | multiline | The formula, evaluated once per position |
| `a`, `b`, `c`… | optional | * | - | - | Lists (or single values); the front end grows one port per letter as they are connected. The port name is the name used in the expression. |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `FLOAT` | FLOAT list | Result of each position as a float |
| `INT` | INT list | Result of each position truncated to an integer |
| `BOOL` | BOOLEAN list | Truthiness of each position's result |

## Alignment

| Input lengths | Result |
| ------------- | ------ |
| All equal, e.g. 3 and 3 | Evaluated pairwise: position 0 with 0, 1 with 1… |
| One input has length 1 | That value is broadcast to every position |
| Single values (not lists) | Treated as length 1 |
| Any other mix, e.g. 3 and 2 | Error: `输入长度不一致`, nothing is evaluated |

## Expression

- Same grammar as `Math Expression`: `+ - * / // % **`, parentheses, comparisons.
- Functions: `sum` `min` `max` `abs` `round` `pow` `sqrt` `ceil` `floor` `log` `log2` `log10` `sin` `cos` `tan` `int` `float`.
- `values` is the list of all inputs at the current position, e.g. `sum(values)`.
- Anything outside this set (imports, attribute access, file access) is rejected by the expression evaluator.
- The result must be numeric. A string result raises an error; a boolean result counts as numeric (`True` = 1).
- `pow` exponents above 4000 are rejected, the same cap as the core node.

## Typical usage

- Add one to every index: `Text Path Extract` (`layers[*].index`) → `a + 1` → `Text Path Replace` (`layers[*].index`).
- Add two lists elementwise: `a = [1, 2, 3]`, `b = [10, 20, 30]`, `a + b` gives `[11, 22, 33]`.
- Scale a list by a single value: `b` with one item, `a * b` multiplies every element by it.

## Notes

- Each port keeps its own letter: if `b` is unconnected, `c` is still `c`, so the expression needs no change.
- The node receives each whole list once (`is_input_list`), so ComfyUI does not run it per element.
- The expression runs without a sandbox beyond `simpleeval`; keep formulas to arithmetic.
