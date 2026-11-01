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

## Never ask questions that are already answered

The user's time is valuable. Asking about something already settled is a serious failure. Before asking anything, check that it is genuinely open. Never ask about:

- Anything decided earlier in the conversation or recorded in memory (e.g., the evaluation protocol, which cell types are used).
- Anything fixed by the reference implementation we follow. Look it up in that code instead of asking (e.g., which Borzoi layer scooby uses as the DNA feature).
- Engineering work that is my own job: data-loader efficiency, batching, memory, and how to implement a design the user specified. Decide it, then report what I did and why.
- Concepts the user did not ask for (e.g., proposing "patchify" when told only to follow the DiT architecture). Implement the user's design as stated. If it has a real blocking problem, state it once, briefly, together with my fix; do not re-litigate the design.

Ask only about genuinely new choices that change the result and cannot be derived from earlier decisions, the reference code, or memory. Keep that list as short as possible.

## Referencing other work: adapt, do not copy

When the user says to refer to an existing model or method (e.g. DiT, Wan, FLUX), they want its relevant design ideas, not a faithful replica.

- Take only the modules and techniques that serve our task (e.g. how Wan applies RoPE, how its blocks are modulated). Drop what does not apply, such as cross-attention, text encoders or timestep encoding when our condition is a single cell vector.
- Sizes, input and output adapters, normalisation, initialisation and anything else specific to our data are ours to design. Base them on our data and task, and explain each adaptation briefly.
- Do not present the reference's details as constraints, and do not turn each of them into a question for the user.

## Other

- Whenever a reply follows code modifications, end it with a short final section listing every code file changed in that turn (created, modified, renamed or deleted), one line per file with what changed. This section is always the last part of the reply.
- Write file contents in English (comments, docs, config files, etc.); conversation with the user may stay in Chinese.
- Before writing, modifying, or reviewing any Python code, invoke the `code-style` skill (once per conversation) and follow its conventions strictly.
