#!/usr/bin/env python3
"""Node metadata consistency checker (single-source-of-truth guard).

nodes/**/*.py is the single source of truth for every node's node_id
and display_name. The README node lists and web/docs/<node_id>/ are
derived views, so this script fails whenever those views drift apart:

* node_id and display_name must be unique across the repository;
* every node must ship web/docs/<node_id>/en.md and .../zh.md and the
  display_name must be mentioned in both;
* web/docs must not keep an orphan directory for a removed node;
* the README node tables must list exactly the current display_name set,
  with no duplicate rows;
* the pyproject version must equal the newest changelog entry in both
  READMEs, so a release has exactly one version authority;
* display_name should follow the spaced model/version convention
  (reported as a warning only, never fatal).

Run from anywhere:

    python scripts/check_node_consistency.py

The exit code is non-zero when a hard error is found, so it can gate CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES_DIR = ROOT / "nodes"
DOCS_DIR = ROOT / "web" / "docs"

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
