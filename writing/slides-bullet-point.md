---
name: slides-bullet-point
description: Convert experiments, approaches, projects, findings, or long technical explanations into concise, presentation-ready bullet-point summaries. Use when the user asks for PPT or slide text, brief experiment summaries, one-line overviews followed by implementation bullets, compact lessons learned, or shorter and more scannable wording.
---

# Slides Bullet Point

Write for a slide, not a report. Preserve the core mechanism and conclusion while removing background, low-level details, and repetition.

## Workflow

1. Identify each distinct experiment, approach, or topic.
2. Extract only its purpose, two or three defining implementation choices, and outcome.
3. Separate observed results from possible explanations.
4. Match the user's requested language and terminology.

## Format

For each topic, use:

```markdown
### [Short Topic Name]

*[One-sentence overview in plain language.]*

- [Core implementation idea.]
- [Second implementation detail.]
- [Optional third detail.]
- **Result:** [Observed outcome; optionally add one brief, qualified explanation.]
```

Use 3–4 bullets by default, including the result. Keep each bullet to one sentence and preferably under 25 words. Start bullets with direct verbs or concrete nouns.

## Writing Rules

- Lead with what was changed, not why the field matters.
- Keep one technical idea per bullet.
- Retain equations or parameter values only when they clarify the defining mechanism.
- Use parallel grammar across related topics.
- Write observed evidence directly: `NMI/ARI decreased`.
- Qualify interpretations: `may`, `likely`, or `suggesting`.
- Avoid unsupported claims, setup narration, code paths, and implementation trivia.
- Avoid nested bullets unless the user explicitly requests more detail.
- If the user asks for shorter text, remove detail before removing the result.

## Example

### Genomic Coordinate Encoding

*Added explicit genomic coordinates to each cCRE token.*

- Learned embeddings encoded chromosome IDs.
- Multi-scale sinusoidal features encoded cCRE midpoint; an MLP encoded interval width.
- Position features were added as `token embedding + learnable scale × position embedding`.
- **Result:** No clear improvement; Borzoi embeddings likely already encode positional context.
