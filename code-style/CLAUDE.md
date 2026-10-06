# Global instructions


## Environment

- The project is on a local Mac which has no Python env or dependencies. Do not run code, tests, or pip install here — only edit and static checks.


## Most important: this is a real production environment (CSCS server cluster)

- Do not modify any code without the user's **explicit** request.
- Do not submit any training or compute jobs (`sbatch` / `srun`, etc.) without the user's **explicit** request, including small "verification" jobs.
- Before modifying, overwriting, or deleting any file, ask the user and wait for confirmation.
- Read-only operations (cat / ls / grep / squeue / sinfo, etc.) may be run directly.
- When making code changes, if verifying them would require running code (tests, smoke tests, scripts, even on CPU), ask the user for consent first. Without consent, only do static checks (syntax parsing, linting, reading the code).

## Decisions belong to the user

- Never treat my own proposal, recommendation, or improvement as a user decision. A point counts as decided only when the user explicitly states it.
- If the user's reply skips or ignores a point, it stays open: neither accepted nor rejected. Ask about it again in the next turn.
- Do not act silently on open points (e.g., baking them into plans, configs, or code as if agreed). When a deliverable depends on an open point, mark it as pending and list it explicitly.

## Other

- Write file contents in English (comments, docs, config files, etc.); conversation with the user may stay in Chinese.
- Before writing, modifying, or reviewing any Python code, invoke the `code-style` skill (once per conversation) and follow its conventions strictly.

## Response Style

- When discussing problems and solutions, users will read every word of your response carefully, so don't post any meaningless drivel that wastes their time.
