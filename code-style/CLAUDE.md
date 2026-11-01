# Global instructions


## Environment: this is a local Mac environment
- No Python env. Never run code, tests, or pip; edit and static checks only.


## Environment: this is a real production environment (CSCS server cluster)
- Read-only commands (cat/ls/grep/squeue/sinfo…) are fine.
- Get explicit user approval before: modifying/overwriting/deleting any file; running any code (tests, smoke tests, scripts, even on CPU); submitting any job (sbatch/srun, even small verification runs). Without approval, static checks only.



## Decisions belong to the user

- Never treat my own proposal, recommendation, or improvement as a user decision. A point counts as decided only when the user explicitly states it.
- If the user's reply skips or ignores a point, it stays open: neither accepted nor rejected. Ask about it again in the next turn.
- Do not act silently on open points (e.g., baking them into plans, configs, or code as if agreed). When a deliverable depends on an open point, mark it as pending and list it explicitly.


## Don't ask what can be looked up
Never ask about anything settled in the conversation or memory, or can be found in the reference implementation (check its code), or irrelevant to our project (e.g., unused parts of the reference).



## Referencing other work: adapt, don't copy
- Take only what serves our task; for each adopted element, explain how it applies to our project.
- Reference details are not constraints. Propose improvements when confident, but discuss them with the user first.


## Other
- File contents in English; conversation may be Chinese.
- Before writing, modifying, or reviewing Python, invoke the `code-style` skill (once per conversation) and follow it strictly.
- After code changes, end the reply with a list of every code file changed that turn (created/modified/renamed/deleted), one line each with the change.

