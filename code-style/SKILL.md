---
name: code-style
description: Enforce Python code style conventions when writing, reviewing, or refactoring Python files. Use when creating new scripts, modifying existing code, or when the user asks to follow project conventions. If this skill has already been loaded in the current conversation, do NOT load it again.
---

# Code Style

## Language

- All code, comments, docstrings, and print/log messages MUST be in English.
- Keep comments minimal. Do NOT add obvious or redundant comments.
- Use `# ---- section title ----` to separate logic blocks in large files.


## Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Class | PascalCase | `VideoSegmenter` |
| Function/Method | snake_case | `build_dataset` |
| Private | `_` prefix | `_preprocess` |
| Constant | UPPER_CASE | `REQUIRED_KEYS` |
| Variable | snake_case | `data_root` |

## Type Hints

All function definitions do NOT need type annotations, example:

```python
def merge_segments(
    segments,
    train_action_chunk,
    ...
):
```

## Formatting

- Indent: 4 spaces
- Trailing comma in multiline list/dict/call
- 2 blank lines between top-level definitions
- 1 blank line between methods inside a class
- If a function/class signature or argument list fits cleanly on one line, prefer single-line over multi-line

## Error Handling
- Do NOT use `try-except` unless absolutely necessary (e.g., parsing untrusted external input).
- Do NOT add defensive checks for invalid inputs unless explicitly required. Let Python raise naturally — a loud error is preferable to silent edge-case handling.

## argparse

- All arguments MUST have default values (no `required=True`).
- Help strings: concise, English only.

```python
parser.add_argument("--root_path", type=str, default="/data/default/path")
parser.add_argument("--train_action_chunk", type=int, default=81)
```
