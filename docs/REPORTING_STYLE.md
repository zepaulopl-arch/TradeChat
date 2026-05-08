# Reporting Style

TradeChat CLI reports now follow a shared terminal presentation layer in `app/presentation.py`.

## Visual rules

- Use muted ANSI colors only for signal direction, positive/negative deltas, and high-salience states.
- Keep section order predictable: header, facts, main table, short notes.
- Prefer one compact table per report instead of multiple noisy blocks.
- Let the terminal width drive density through `screen_width()` and compact column sets.

## Layout primitives

- `banner(...)` for the title block
- `render_facts(...)` for key-value summaries
- `render_table(...)` for aligned tables
- `render_wrapped(...)` for long notes and file paths
- `divider(...)` only between major sections

## Information policy

- Show only decision-relevant numbers by default.
- Demote operational detail to wrapped notes or the text audit report.
- Use compact mode to drop secondary columns instead of breaking alignment.
- Keep text reports plain ASCII with no ANSI color codes.
