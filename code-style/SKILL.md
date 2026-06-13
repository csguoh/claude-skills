---
name: code-style
description: Enforce Python code style conventions when writing, reviewing, or refactoring Python files. Use when creating new scripts, modifying existing code, or when the user asks to follow project conventions. If this skill has already been loaded in the current conversation, do NOT load it again.
---

# Code Style

## Language

- All code, comments, docstrings, and print/log messages MUST be in English.
- Keep comments minimal. Do NOT add obvious or redundant comments.
- When you believe a comment is necessary, use one of these two forms:
  - **Logic block separators** (`""" ---- section title ---- """`): used to visually split large files into sections. Surround with blank lines above and below.
  - **Function/line annotations** (`# xxx`): a short plain comment describing what the next function or line does. No blank line between this comment and the code immediately below it. Format is just `# xxx` (no dashes, no decoration).
- Do NOT add a file-level docstring or comment describing what the file does at the top.

## Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Class | PascalCase | `VideoSegmenter` |
| Function/Method | snake_case | `build_dataset` |
| Private | `__` prefix | `_preprocess` |
| Constant | UPPER_CASE | `REQUIRED_KEYS` |
| Variable | snake_case | `data_root` |

> Function/method names must be descriptive (2–4 words), e.g. `_build_video_dataset`, `_parse_action_chunk`. Avoid vague single-word names like `_build` or `_parse` — they cause name collisions across classes and are unreadable without context.

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

## Brevity Principle

- Prefer concise code over defensive code. Make strong format assumptions on external data (e.g., assume inputs are always a list of dicts with specific keys) rather than adding layers of `if/else` or type-checking. Add a short comment stating the expected input format instead.
- Never write branches that are unlikely to be reached in practice — a natural Python error is more informative than dead code.
- When in doubt, let it crash. Short, readable code that fails loudly on bad input is better than bloated code that silently handles imaginary edge cases.

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
