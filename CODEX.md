# Codex Working Agreement (nano-train)

This file defines the expected coding and writing style for changes made with Codex.

## Coding Style

Follow `AGENTS.md` (line length 100, explicit logic, tests, readable code, docstrings for public APIs).

## Documentation Writing Style (Reports, Docs, Design Notes)

When writing Markdown in this repo (including generated reports), use a top-systems-paper style:
clear narrative, explicit definitions, and auditable reasoning.

- Prefer coherent paragraphs over bullet checklists. Use bullets only for true enumerations.
- Use “We …” voice in reports to explain intent and reasoning (e.g., “We define …”, “We estimate …”).
- Start each major section with a short bridge paragraph that answers:
  what this section does, why it exists, where the evidence is (table/figure), and what it does not claim.
- Term hygiene:
  define all acronyms/symbols before first use; keep notation consistent; include units.
  If a term is used repeatedly, add a small glossary table near the top.
- Make the reasoning chain explicit and stable:
  `inputs/assumptions -> FLOPs/bytes -> AI -> time model -> plotted/summary metrics`.
  Clearly label *upper bounds* (roofline ceilings) vs *estimates* (time-model throughput).
- Tables/figures:
  introduce each table/figure with 2–4 sentences explaining what it shows and how to read it.
  Keep plots uncluttered; put detailed numbers in tables; include key config in captions/titles.
- Avoid over-claiming:
  static models are not measured performance; separate algorithmic FLOPs from realizable/achievable FLOPs.
- Editing discipline:
  trim repetition, keep definitions in one canonical place, and avoid mixing alternative notations.

