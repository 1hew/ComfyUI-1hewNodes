#!/usr/bin/env python3
"""Node metadata consistency checker (single-source-of-truth guard).

nodes/**/*.py is the single source of truth for every node's node_id
and display_name. The README node lists and web/docs/<node_id>/ are
derived views, so this script fails whenever those views drift apart:

* every module under nodes/ must follow the one-node-per-file layout:
  a nodes/<category>/*.py (and any nodes/*.py without a leading
  underscore) defines exactly one node, while shared helpers live in
  utils/_*.py at the package root;
* no file may import a node module from another category, so a shared
  helper cannot hide inside an unrelated node file;
* node_id and display_name must be unique across the repository;
* every node must ship web/docs/<node_id>/en.md and .../zh.md and the
  display_name must be mentioned in both;
* web/docs must not keep an orphan directory for a removed node;
* the README node tables must list exactly the current display_name set,
  with no duplicate rows;
* the pyproject version must equal the newest changelog entry in both
  READMEs, so a release has exactly one version authority;
* every web/js/dynamic_port.js entry must name a current node_id whose
  declared ports match the configured base / addType / select / output,
  so the front-end dynamic-port table cannot drift from the schemas;
* every quoted 1hew_* literal in web/**/*.js must be a current node_id,
  so a renamed or removed node cannot leave a stale front-end reference;
* the Save Text metadata markers in utils/_text_metadata.py and
  web/js/core/text_metadata.js must stay identical, so backend-written
  blocks are always recognized by the front-end drop handler;
* display_name should follow the spaced model/version convention
  (reported as a warning only, never fatal).

Run from anywhere:

    python scripts/check_node_consistency.py

The exit code is non-zero when a hard error is found, so it can gate CI.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES_DIR = ROOT / "nodes"
WEB_DIR = ROOT / "web"
DOCS_DIR = WEB_DIR / "docs"

# README node-list section heading -> file
READMES = {
    "README.md": "## \U0001F4CB Node List",
    "README.ZH_CN.md": "## \U0001F4CB \u8282\u70b9\u5217\u8868",
}

# README changelog section heading -> file
CHANGELOGS = {
    "README.md": "## \U0001F4DC Changelog",
    "README.ZH_CN.md": "## \U0001F4DC \u66f4\u65b0\u65e5\u5fd7",
}

PYPROJECT = ROOT / "pyproject.toml"

NODE_ID_RE = re.compile(r'node_id="([^"]+)"')
DISPLAY_NAME_RE = re.compile(r'display_name="([^"]+)"')
SEPARATOR_RE = re.compile(r":?-{2,}:?")
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
CHANGELOG_VERSION_RE = re.compile(r"^\*\*v([0-9][0-9A-Za-z.\-]*)\*\*$")

# Front-end dynamic-port table: node_id -> {base, addType, select, initial, ...}
DYNAMIC_PORTS_FILE = ROOT / "web" / "js" / "dynamic_port.js"
TEXT_METADATA_PY = ROOT / "utils" / "_text_metadata.py"
TEXT_METADATA_JS = ROOT / "web" / "js" / "core" / "text_metadata.js"
PY_METADATA_MARKER_RE = re.compile(
    r'^(METADATA_(BEGIN|END))\s*=\s*"([^"]+)"', re.MULTILINE
)
JS_METADATA_MARKER_RE = re.compile(
    r'export const TEXT_METADATA_(BEGIN|END)\s*=\s*"([^"]+)"'
)
DYNAMIC_CONFIG_ENTRY_RE = re.compile(r'"(1hew_[A-Za-z0-9_]+)"\s*:\s*\{([^}]*)\}')
DYNAMIC_CONFIG_FIELD_RE = re.compile(r'(\w+)\s*:\s*(?:"([^"]*)"|(\d+)|(null))')
SCHEMA_INPUT_RE = re.compile(r'io\.(?:Custom\("([^"]+)"\)|([A-Za-z]+))\.Input\(\s*(f?)"([^"]*)"')
SCHEMA_OUTPUT_RE = re.compile(r'io\.(?:Custom\("([^"]+)"\)|([A-Za-z]+))\.Output\(\s*display_name=(f?)"([^"]*)"')

# Any quoted 1hew_* literal in the front-end must be a current node_id.
# Internal property names (e.g. node.1hew_step_config) are unquoted, so a
# quoted literal is always a node reference.
JS_NODE_ID_RE = re.compile(r"""["'](1hew_[A-Za-z0-9_]+)["']""")

# io.<Builtin>.Input -> the LiteGraph port type dynamic_port.js adds at runtime
PORT_TYPE_MAP = {
    "Image": "IMAGE",
    "Mask": "MASK",
    "Video": "VIDEO",
    "Audio": "AUDIO",
    "String": "STRING",
    "Int": "INT",
    "Float": "FLOAT",
    "Boolean": "BOOLEAN",
    "Combo": "COMBO",
}

# display_name patterns that violate the spaced naming convention
CAMEL_RE = re.compile(r"[a-z][A-Z]")
GLUED_VERSION_RE = re.compile(r"[A-Za-z]\d")

# Intentional compact names that are not model+version tokens. Keep this list
# explicit so every exception is a conscious decision, not a silent oversight.
COMPACT_NAME_ALLOWLIST = {
    "Image PingPong",    # compound word, not a model name
    "Mask to SAM3 Box",  # "SAM3" is the model's common compact spelling
}


def collect_nodes():
    """Return (node_id, display_name, path) for every node definition."""
    nodes = []
    for path in sorted(NODES_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        id_match = NODE_ID_RE.search(text)
        if not id_match:
            continue
        # Only accept a display_name from the same Schema block, i.e. the
        # first one after node_id= (this skips IO port display names).
        name_match = DISPLAY_NAME_RE.search(text, id_match.end())
        if not name_match:
            continue
        nodes.append((id_match.group(1), name_match.group(1), path))
    return nodes


def table_names(readme, heading):
    """Return the first column of every data row in the node-list section."""
    names = []
    in_section = False
    for line in readme.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == heading
            continue
        if not in_section or not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.split("|")]
        if len(cells) < 4:
            continue
        name = cells[1]
        if not name or name in {"Node Name", "\u8282\u70b9\u540d\u79f0"} or SEPARATOR_RE.fullmatch(name):
            continue
        names.append(name)
    return names


def top_changelog_version(readme, heading):
    """Return the newest version in the README changelog section."""
    in_section = False
    for line in readme.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == heading
            continue
        if not in_section:
            continue
        match = CHANGELOG_VERSION_RE.match(stripped)
        if match:
            return match.group(1)
    return None


def port_type(custom, builtin):
    """Resolve an io.Custom(X) or io.X input type to its LiteGraph port name."""
    return custom if custom else PORT_TYPE_MAP.get(builtin, builtin)


def schema_ports(text, pattern):
    """Return (type, is_fstring, name) for every matching IO declaration."""
    return [
        (port_type(match.group(1), match.group(2)), match.group(3) == "f", match.group(4))
        for match in pattern.finditer(text)
    ]


def dynamic_configs():
    """Parse web/js/dynamic_port.js into ({node_id: fields}, [duplicate ids])."""
    configs = {}
    duplicates = []
    text = DYNAMIC_PORTS_FILE.read_text(encoding="utf-8")
    for match in DYNAMIC_CONFIG_ENTRY_RE.finditer(text):
        node_id, body = match.group(1), match.group(2)
        if node_id in configs:
            duplicates.append(node_id)
        fields = {}
        for field in DYNAMIC_CONFIG_FIELD_RE.finditer(body):
            if field.group(2) is not None:
                value = field.group(2)
            elif field.group(3) is not None:
                value = int(field.group(3))
            else:
                value = None
            fields[field.group(1)] = value
        configs[node_id] = fields
    return configs, duplicates


def check_dynamic_ports(nodes_by_id, errors, warnings):
    """Fail when the front-end dynamic-port table drifts from the node schemas."""
    configs, duplicates = dynamic_configs()
    for node_id in duplicates:
        errors.append("dynamic_port.js has duplicate config key: %s" % node_id)

    for node_id, cfg in sorted(configs.items()):
        if node_id not in nodes_by_id:
            errors.append("dynamic_port.js references unknown node_id: %s" % node_id)
            continue
        text = nodes_by_id[node_id].read_text(encoding="utf-8")
        inputs = schema_ports(text, SCHEMA_INPUT_RE)
        outputs = schema_ports(text, SCHEMA_OUTPUT_RE)

        base = cfg.get("base")
        add_type = cfg.get("addType")
        if not isinstance(base, str) or not base:
            errors.append("%s: dynamic_port.js base must be a non-empty string" % node_id)
            continue
        if not isinstance(add_type, str) or not add_type:
            errors.append("%s: dynamic_port.js addType must be a non-empty string" % node_id)

        base_inputs = [port for port in inputs if port[2].startswith(base)]
        if not base_inputs:
            declared = ", ".join(sorted({port[2] for port in inputs})) or "none"
            errors.append(
                "%s: dynamic_port.js base %r matches no declared input (declared: %s)"
                % (node_id, base, declared)
            )
        for ptype, _is_fstring, pname in base_inputs:
            if add_type and ptype != add_type:
                errors.append(
                    "%s: dynamic_port.js addType %r != declared type %r of input %r"
                    % (node_id, add_type, ptype, pname)
                )

        select = cfg.get("select")
        if select and select not in {port[2] for port in inputs}:
            errors.append(
                "%s: dynamic_port.js select %r matches no declared input" % (node_id, select)
            )

        output_base = cfg.get("outputBase")
        output_type = cfg.get("outputType")
        if output_base:
            base_outputs = [port for port in outputs if port[2].startswith(output_base)]
            if not base_outputs:
                errors.append(
                    "%s: dynamic_port.js outputBase %r matches no declared output"
                    % (node_id, output_base)
                )
            for ptype, _is_fstring, pname in base_outputs:
                if output_type and ptype != output_type:
                    errors.append(
                        "%s: dynamic_port.js outputType %r != declared type %r of output %r"
                        % (node_id, output_type, ptype, pname)
                    )

        initial = cfg.get("initial")
        cap = cfg.get("max")
        if not isinstance(initial, int) or initial < 1:
            errors.append("%s: dynamic_port.js initial must be a positive integer" % node_id)
        if cap is not None:
            if not isinstance(cap, int) or cap < 1:
                errors.append("%s: dynamic_port.js max must be a positive integer" % node_id)
            elif isinstance(initial, int) and initial > cap:
                errors.append(
                    "%s: dynamic_port.js initial (%s) exceeds max (%s)"
                    % (node_id, initial, cap)
                )

        explicit = [port for port in base_inputs if not port[1]]
        if isinstance(initial, int) and initial != len(explicit):
            warnings.append(
                "%s: dynamic_port.js initial=%s but the node declares %d explicit %r input(s)"
                % (node_id, initial, len(explicit), base)
            )


def check_js_node_ids(known_ids, errors):
    """Fail when web/js references a node_id that no node defines."""
    for path in sorted(WEB_DIR.rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        for match in JS_NODE_ID_RE.finditer(text):
            token = match.group(1)
            if token not in known_ids:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    "%s:%d references unknown node_id: %s"
                    % (path.relative_to(ROOT), line, token)
                )


def check_text_metadata_markers(errors):
    """The backend writer and the front-end reader must share markers."""
    py_markers = {
        match.group(1): match.group(3)
        for match in PY_METADATA_MARKER_RE.finditer(
            TEXT_METADATA_PY.read_text(encoding="utf-8")
        )
    }
    js_markers = {
        "METADATA_" + match.group(1): match.group(2)
        for match in JS_METADATA_MARKER_RE.finditer(
            TEXT_METADATA_JS.read_text(encoding="utf-8")
        )
    }
    for name in ("METADATA_BEGIN", "METADATA_END"):
        py_value = py_markers.get(name)
        js_value = js_markers.get(name)
        if py_value is None or js_value is None:
            errors.append(
                "Save Text metadata marker %s is missing from the Python "
                "writer or the front-end reader" % name
            )
        elif py_value != js_value:
            errors.append(
                "Save Text metadata marker %s differs: Python %r != JS %r"
                % (name, py_value, js_value)
            )


LIFECYCLE_METHODS = {
    "define_schema",
    "define_node",
    "execute",
    "validate_inputs",
    "fingerprint_inputs",
    "IS_CHANGED",
    "__init__",
    "unload",
}
DUPLICATE_HELPER_MIN_LINES = 20


def check_node_file_layout(errors):
    """Every node module defines exactly one node; helpers live in utils/_*.py."""
    for path in sorted(NODES_DIR.rglob("*.py")):
        rel = path.relative_to(NODES_DIR)
        if "__pycache__" in rel.parts or path.name == "__init__.py":
            continue
        if len(rel.parts) == 1 and path.name.startswith("_"):
            continue
        count = len(NODE_ID_RE.findall(path.read_text(encoding="utf-8")))
        if count != 1:
            errors.append(
                "nodes/%s defines %d node(s); exactly one node per file is required "
                "(shared helpers belong in utils/_*.py)" % (rel.as_posix(), count)
            )


def _resolve_relative_import(importer, level, module):
    target_dir = importer.parent
    for _ in range(level - 1):
        target_dir = target_dir.parent
    return target_dir / module.replace(".", "/") if module else target_dir


def check_node_module_imports(errors):
    """A node module must never be imported from a different category."""
    for path in sorted(NODES_DIR.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = _resolve_relative_import(path, node.level, node.module)
            target = base.with_suffix(".py")
            if not target.is_file() or target == path:
                continue
            if not NODE_ID_RE.search(target.read_text(encoding="utf-8")):
                continue
            if target.parent == path.parent:
                continue
            errors.append(
                "%s:%d imports node module %s from another category; move the "
                "shared helper into utils/_*.py"
                % (path.relative_to(ROOT), node.lineno, target.relative_to(ROOT))
            )


def check_duplicate_helpers(warnings):
    """Warn when the same long helper body is copied into several files."""
    seen = {}
    for path in sorted(NODES_DIR.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in LIFECYCLE_METHODS:
                continue
            size = node.end_lineno - node.lineno + 1
            if size < DUPLICATE_HELPER_MIN_LINES:
                continue
            digest = hashlib.sha1(
                ast.dump(ast.Module(body=node.body, type_ignores=[])).encode()
            ).hexdigest()
            seen.setdefault(digest, []).append((path.relative_to(ROOT), node.name, size))
    for group in seen.values():
        if len(group) < 2:
            continue
        first = group[0]
        warnings.append(
            "%s is duplicated (%d lines) in %s; extract it into utils/_*.py"
            % (first[1], first[2], ", ".join(str(p) for p, _, _ in group))
        )


def main():
    errors = []
    warnings = []

    nodes = collect_nodes()
    ids = [node_id for node_id, _, _ in nodes]
    names = [display_name for _, display_name, _ in nodes]

    for label, values in (("node_id", ids), ("display_name", names)):
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        for value, count in sorted(counts.items()):
            if count > 1:
                errors.append("duplicate %s: %r (%dx)" % (label, value, count))

    for node_id, display_name, path in nodes:
        doc_dir = DOCS_DIR / node_id
        for locale in ("en.md", "zh.md"):
            doc = doc_dir / locale
            if not doc.is_file():
                errors.append(
                    "missing docs: %s (node %s)"
                    % (doc.relative_to(ROOT), path.relative_to(ROOT))
                )
            elif display_name not in doc.read_text(encoding="utf-8"):
                errors.append(
                    "docs %s does not mention display_name %r"
                    % (doc.relative_to(ROOT), display_name)
                )

    for readme_name, heading in READMES.items():
        readme = ROOT / readme_name
        listed = table_names(readme, heading)
        missing = sorted(set(names) - set(listed))
        unknown = sorted(set(listed) - set(names))
        if missing:
            errors.append("%s node table is missing: %s" % (readme_name, ", ".join(missing)))
        if unknown:
            errors.append(
                "%s node table lists unknown nodes: %s" % (readme_name, ", ".join(unknown))
            )
        if len(listed) != len(set(listed)):
            duplicates = sorted({name for name in listed if listed.count(name) > 1})
            errors.append(
                "%s node table has duplicate rows: %s"
                % (readme_name, ", ".join(duplicates))
            )

    known_ids = set(ids)
    for doc_dir in sorted(DOCS_DIR.iterdir()):
        if doc_dir.is_dir() and doc_dir.name not in known_ids:
            errors.append(
                "orphan docs directory: %s (no node defines it)"
                % doc_dir.relative_to(ROOT)
            )

    version_match = VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if version_match is None:
        errors.append("pyproject.toml has no version")
    else:
        version = version_match.group(1)
        for readme_name, heading in CHANGELOGS.items():
            top = top_changelog_version(ROOT / readme_name, heading)
            if top is None:
                errors.append("%s changelog has no versioned entry" % readme_name)
            elif top != version:
                errors.append(
                    "%s newest changelog v%s != pyproject version %s"
                    % (readme_name, top, version)
                )

    nodes_by_id = {node_id: path for node_id, _, path in nodes}
    check_dynamic_ports(nodes_by_id, errors, warnings)
    check_js_node_ids(known_ids, errors)
    check_text_metadata_markers(errors)
    check_node_file_layout(errors)
    check_node_module_imports(errors)
    check_duplicate_helpers(warnings)

    for _, display_name, _ in nodes:
        if display_name in COMPACT_NAME_ALLOWLIST:
            continue
        if CAMEL_RE.search(display_name):
            warnings.append("%r contains a camelCase token" % display_name)
        if GLUED_VERSION_RE.search(display_name):
            warnings.append("%r glues a version number to letters" % display_name)

    print("nodes: %d" % len(nodes))
    for warning in warnings:
        print("WARN  " + warning)
    for error in errors:
        print("ERROR " + error)
    print("OK" if not errors else "FAILED (%d error(s))" % len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
