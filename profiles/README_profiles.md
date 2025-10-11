# Profiles Exemplars Guidelines

Use this directory to organize retrieval-ready exemplars, glossaries, and style notes for each theme. Follow these rules to keep Stage 2 plug-and-play:

1. **Chunk size**: each exemplar file should contain 200–700 words focused on a single idea. Avoid multi-topic dumps.
2. **File naming**: use zero-padded numeric prefixes so ordering is explicit, e.g. `001_intro.md`, `002_benefits.md`, `003_faq.md`.
3. **Content style**: prefer light Markdown (headings, lists) only when it improves clarity. Heavy formatting, tables, or embeds are discouraged.
4. **PII policy**: do not store personally identifiable information or sensitive internal data.
5. **Glossary**: include theme-specific terminology in `glossary.txt` with short definitions (one term per line: `term — description`).
6. **Style guide**: describe tone, voice, and formatting do's/don'ts in `style_guide.md`. Mention heading usage, bullet preferences, and localization needs.
7. **Quality check**: ensure language aligns with base prompt expectations (Russian by default) and that examples are factual and timeless.

When adding new themes, copy this structure:

```
profiles/<theme>/
  exemplars/
    001_intro.md
    002_benefits.md
    003_faq.md
  glossary.txt
  style_guide.md
```

Keep exemplar content evergreen so it can safely influence future generations.
