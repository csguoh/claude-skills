---
name: code-style
description: Enforce Python code style conventions when writing, reviewing, or refactoring Python files. Use when creating new scripts, modifying existing code, or when the user asks to follow project conventions. If this skill has already been loaded in the current conversation, do NOT load it again.
---

# Code Style

## Language

- All code, comments, docstrings, and print/log messages MUST be in English.
- Keep comments minimal. Do NOT add obvious or redundant comments.
- Use `# ---- section title ----` to separate logic blocks in large files.
- No blank line between a `# ---- ... ----` section comment and the function/class definition immediately below it.
- Do NOT add a file-level docstring or comment describing what the file does at the top.

## Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Class | PascalCase | `VideoSegmenter` |
| Function/Method | snake_case | `build_dataset` |
| Private | `__` prefix | `_preprocess` |
| Constant | UPPER_CASE | `REQUIRED_KEYS` |
| Variable | snake_case | `data_root` |

> Function/method names must be descriptive (3–4 words), e.g. `_build_video_dataset`, `_parse_action_chunk`. Avoid vague single-word names like `_build` or `_parse` — they cause name collisions across classes and are unreadable without context.

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
