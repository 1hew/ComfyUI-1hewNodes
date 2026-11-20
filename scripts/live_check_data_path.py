"""Live smoke test against a running ComfyUI.

Submits small prompts through the HTTP API and compares the executed outputs
with what the nodes promise in their docs. PreviewAny is used as the sink so
the outputs show up in /history without writing files.

    python scripts/live_check_data_path.py [base_url]
"""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

API = "http://127.0.0.1:9627"


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} from {path}:", file=sys.stderr)
        print(exc.read().decode("utf-8", "replace")[:2000], file=sys.stderr)
        raise


def _get(path):
    with urllib.request.urlopen(API + path) as response:
        return json.load(response)


def run(graph, timeout=120):
    out = _post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    pid = out["prompt_id"]
    for _ in range(timeout * 2):
        time.sleep(0.5)
        history = _get(f"/history/{pid}")
        if pid in history:
            entry = history[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    "execution error: "
                    + json.dumps(status.get("messages", []), ensure_ascii=False)[:1500]
                )
            return entry["outputs"]
    raise TimeoutError(pid)


def sink(graph, source, output_index, name="preview"):
    """Attach a PreviewAny sink to graph[source] output index."""
    graph[name] = {
        "class_type": "PreviewAny",
        "inputs": {"source": [source, output_index]},
    }
    return name


def const(graph, value, name="const"):
    """A wildcard port cannot take a literal, so feed it through Text To Any."""
    graph[name] = {
        "class_type": "1hew_TextToAny",
        "inputs": {"text": json.dumps(value, ensure_ascii=False)},
    }
    return name


def shown(outputs, node_id):
    """PreviewAny reports its value as a list of text items."""
    return outputs.get(node_id, {}).get("text")


def check(label, actual, expected):
    ok = actual == expected
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        print("       expected:", json.dumps(expected, ensure_ascii=False))
        print("       actual  :", json.dumps(actual, ensure_ascii=False))
    return ok


SAMPLE = '{"layers":[{"index":1},{"index":2},{"index":3}]}'


def extract_graph(data_source, data_path, sink_at="preview"):
    """Data Path Extract takes its payload on a wildcard link, not a literal."""
    graph = {
        "src": {"class_type": "1hew_TextToAny", "inputs": {"text": data_source}},
        "1": {
            "class_type": "1hew_DataPathExtract",
            "inputs": {"data": ["src", 0], "data_path": data_path},
        },
    }
    sink(graph, "1", 0, sink_at)
    return graph


def main():
    results = []

    # 1. Extract: one list item per match, on the value port.
    graph = extract_graph(SAMPLE, "layers[*].index")
    results.append(
        check("Extract value -> ['1','2','3']", shown(run(graph), "preview"), ["1", "2", "3"])
    )

    # 2. Extract: a single match still arrives as a list.
    graph = extract_graph(SAMPLE, "layers[0].index")
    results.append(check("Extract single value -> ['1']", shown(run(graph), "preview"), ["1"]))

    # 3. Extract: a miss gives an empty list and a zero count. PreviewAny
    # renders an empty list as "None", so the count port is the reliable check.
    graph = extract_graph(SAMPLE, "nothing[*].index", "count")
    sink(graph, "1", 1, "count")
    results.append(check("Extract miss count -> ['0']", shown(run(graph), "count"), ["0"]))

    # 4. List Math Expression: element-wise a + 1 on the INT port.
    graph = {
        "src": {"class_type": "1hew_TextToAny", "inputs": {"text": SAMPLE}},
        "1": {
            "class_type": "1hew_DataPathExtract",
            "inputs": {"data": ["src", 0], "data_path": "layers[*].index"},
        },
        "3": {
            "class_type": "1hew_ListMathExpression",
            "inputs": {"expression": "a + 1", "a": ["1", 0]},
        },
    }
    sink(graph, "3", 0, "float")
    sink(graph, "3", 1, "int")
    outputs = run(graph)
    results.append(check("List Math FLOAT -> ['2.0','3.0','4.0']", shown(outputs, "float"), ["2.0", "3.0", "4.0"]))
    results.append(check("List Math INT -> ['2','3','4']", shown(outputs, "int"), ["2", "3", "4"]))

    # 5. Replace writes one typed value, broadcast to every match.
    graph = {
        "src": {"class_type": "1hew_TextToAny", "inputs": {"text": SAMPLE}},
        "val": {"class_type": "1hew_TextToAny", "inputs": {"text": "9"}},
        "4": {
            "class_type": "1hew_DataPathReplace",
            "inputs": {
                "data": ["src", 0],
                "data_path": "layers[*].index",
                "value": ["val", 0],
                "value_type": "int",
            },
        },
    }
    sink(graph, "4", 0, "chain_value")
    outputs = run(graph)
    chain = shown(outputs, "chain_value")
    # The data port keeps the input shape, so PreviewAny shows the object here.
    results.append(
        check(
            "Replace broadcasts one typed value",
            json.loads(chain[0]) if chain else None,
            {"layers": [{"index": 9}, {"index": 9}, {"index": 9}]},
        )
    )

    # 6. List Text: prefix and suffix per item, still a list.
    graph = {
        "src": {"class_type": "1hew_TextToAny", "inputs": {"text": '["cat","dog"]'}},
        "1": {
            "class_type": "1hew_DataPathExtract",
            "inputs": {"data": ["src", 0], "data_path": "[*]"},
        },
        "5": {
            "class_type": "1hew_ListText",
            "inputs": {"prefix": "a ", "suffix": "!", "text_list": ["1", 0]},
        },
    }
    sink(graph, "5", 0, "listtext")
    results.append(
        check("List Text -> ['a cat!','a dog!']", shown(run(graph), "listtext"), ["a cat!", "a dog!"])
    )

    # 7. Replace: broadcast a text value to every match.
    graph = {
        "src": {
            "class_type": "1hew_TextToAny",
            "inputs": {"text": '[{"content":"a"},{"content":"b"}]'},
        },
        "val": {"class_type": "1hew_TextToAny", "inputs": {"text": "X"}},
        "4": {
            "class_type": "1hew_DataPathReplace",
            "inputs": {
                "data": ["src", 0],
                "data_path": "[*].content",
                "value": ["val", 0],
                "value_type": "string",
            },
        },
    }
    sink(graph, "4", 0, "broadcast")
    broadcast = shown(run(graph), "broadcast")
    results.append(
        check(
            "Replace broadcast",
            json.loads(broadcast[0]) if broadcast else None,
            [{"content": "X"}, {"content": "X"}],
        )
    )

    passed = sum(1 for r in results if r)
    print()
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        API = sys.argv[1].rstrip("/")
    sys.exit(main())
