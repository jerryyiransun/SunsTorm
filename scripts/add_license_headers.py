#!/usr/bin/env python3

from pathlib import Path

YEAR = "2026"
OWNER = "Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors"

CPP_HEADER = f"""/*
 * Copyright {YEAR} {OWNER}
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */
"""

PY_HEADER = f"""# Copyright {YEAR} {OWNER}
#
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""

CPP_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp"}
PY_EXTS = {".py", ".sh"}

SKIP_DIRS = {
    ".git",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "__pycache__",
}

SKIP_FILES = {
    "mlsys",  # compiled binary
    "writeup.pdf",
    "LICENSE",
}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.name in SKIP_FILES:
        return True
    if path.suffix not in CPP_EXTS and path.suffix not in PY_EXTS:
        return True
    return False


def already_has_license(text: str) -> bool:
    first_chunk = text[:1000].lower()
    return (
        "apache license" in first_chunk
        or "licensed under the apache license" in first_chunk
        or "copyright" in first_chunk
    )


def add_header(path: Path):
    text = path.read_text(encoding="utf-8")

    if already_has_license(text):
        print(f"skip: {path}")
        return

    if path.suffix in CPP_EXTS:
        new_text = CPP_HEADER + "\n" + text
    elif path.suffix in PY_EXTS:
        if text.startswith("#!"):
            first_line, rest = text.split("\n", 1)
            new_text = first_line + "\n" + PY_HEADER + "\n" + rest
        else:
            new_text = PY_HEADER + "\n" + text
    else:
        return

    path.write_text(new_text, encoding="utf-8")
    print(f"updated: {path}")


def main():
    root = Path(".")
    for path in root.rglob("*"):
        if path.is_file() and not should_skip(path):
            add_header(path)


if __name__ == "__main__":
    main()
