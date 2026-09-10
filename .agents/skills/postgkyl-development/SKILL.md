---
name: postgkyl-development
description: Set up Postgkyl development, choose verification commands, or maintain shared agent configuration and Entire hooks.
---

# Develop and verify

Install NumPy before the extension so it builds against the runtime ABI:

```bash
pip install --upgrade numpy setuptools wheel
pip install --no-build-isolation -e '.[test]'
```

Run focused tests for the change, then the required broader checks:

```bash
pytest tests/
# Without an editable install:
PYTHONPATH=src python -m pytest tests/
```

Useful API/CLI smoke checks (replace data filenames with suitable fixtures):

```bash
pgkyl --help
pgkyl --version
pgkyl file.gkyl info
pgkyl file.gkyl interpolate select --z0 0 plot
pgkyl euler_5m_0.gkyl interpolate five_moment_pressure --num_moms 5 plot
pgkyl a.gkyl b.gkyl evaluate "f0 f1 +" interpolate plot
```

Shared agent configuration lives in `.agents/`; `.claude` and `.codex` are
relative symlinks to it. Root `CLAUDE.md` links to `AGENTS.md`. Put focused skills
in `.agents/skills/<name>/SKILL.md` with name and description frontmatter. Keep
root instructions short and route only to skills relevant to the task.

Preserve Entire's Claude hooks in `settings.json` and Codex hooks in `hooks.json`
inside the shared directory, including matchers, commands, and timeouts. Their
original `.claude/settings.json` and `.codex/hooks.json` paths must still resolve.
Leave `.entire/` and Entire's Git hooks intact. Retain existing agent definitions
and historical plans when reorganizing folders; historical markdown remains
ignored, while new skill entry points must be visible to Git.

For instruction/configuration-only changes, validate skill frontmatter, relative
links, Git visibility, and hook preservation; running scientific tests is not
necessary unless application behavior changes.
