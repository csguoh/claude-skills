---
name: machine-learning-paper-writing
description: Draft, revise, and reorganize computer-vision and machine-learning research papers in LaTeX according to the user's preferred academic style. Use when writing or editing abstracts, introductions, related work, motivation, methods, experiments, discussions, conclusions, captions, equations, tables, appendices, or terminology.
---

# Machine Learning Paper Writing

## Core Writing Style

- Write paper content in clear academic English and discuss decisions with the user in Chinese unless requested otherwise.
- Prefer simple words, short sentences, and direct subject--verb--object constructions. Avoid nested clauses, inflated vocabulary, and unnecessarily formal phrasing.
- Preserve the language style and level of detail already established in the paper. Revise locally unless the user explicitly requests a rewrite.
- State the main point early. Make every following sentence explain its motivation, mechanism, evidence, or consequence.
- Build an explicit logical chain between adjacent sentences. Add a short transition whenever the next sentence changes from motivation to design, from global to local information, or from an observation to its solution.
- Explain why a design is needed before describing how it works. Then state what the resulting component provides.
- Do not overstate results. Connect performance claims to the tasks and evidence actually reported.

## Reasoning and Paragraph Organization

- Organize a technical paragraph as: context or limitation, motivation, proposed operation, and resulting benefit.
- Organize an experimental observation as: experimental setup, observed result, interpretation, and conclusion.
- Introduce a subsection with one or two orienting sentences before enumerating challenges or components.
- Keep one coherent topic per paragraph. Do not split a short, tightly connected argument into multiple paragraphs.
- Avoid abrupt transitions such as moving from a global representation to a stage feature without explaining why stage adaptation is necessary.

## Abstract and Introduction

- Treat the abstract as a compressed version of the introduction. Allocate one or two sentences to each essential part: background, problem, insight, method, extension, and results.
- Do not add code or project links unless requested.
- Develop the introduction in a clean sequence: recent progress, unresolved problems, intuitive solutions, proposed method and results, contributions.
- Describe each challenge separately and concisely. 

## Related Work

- Keep each related-work subsection as one paragraph unless the user requests otherwise.
- Never invent a citation. Reuse citations already present in the project or explicitly supplied by the user. If a needed reference is unavailable and searching is not authorized, insert `TODO(method name)` as a visible placeholder.

## Motivation and Method Writing

- Separate preliminaries, challenges, experimental observations, design insights, architecture overview, and technical modules at appropriate subsection levels.
- Cross-reference the relevant pipeline or motivation subfigure when it supplies visual evidence or presents a module structure.
- Introduce a module by stating the problem it addresses, naming the module, and then explaining its operations in computation order.

## Equations and Symbols

- Minimize displayed equations. Use them only for core method steps or concise summaries that materially improve understanding.
- Keep the number of variables small. Reuse established symbols and verify that every new symbol is necessary.
- Define important tensor shapes near first use. Omit the batch dimension throughout unless it is technically essential.
- Align multi-line equations cleanly. 
- Explain each non-obvious symbol and shape after the equation, but do not paraphrase every obvious arithmetic operation.

## Terminology and Naming

- Treat module names and acronyms as global interfaces. When renaming one, update titles, prose, equations, captions, tables, figure references, contribution lists, appendices, and abbreviations throughout the project.
- Search for old names, old acronyms, spelling variants, and related labels after every rename.

## Tables, Figures, and Experiments

- Transfer reported values exactly. Leave the proposed method's cells empty when the user has not supplied results.
- Move a figure, table, caption, label, and its discussion together when reorganizing sections.
