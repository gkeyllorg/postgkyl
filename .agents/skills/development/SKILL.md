---
name: development
description: Set up Postgkyl development, format code, choose verification commands, or maintain shared agent configuration and Entire hooks.
---

# Develop and verify

Install NumPy before the extension so it builds against the runtime ABI:

```bash
pip install --upgrade numpy setuptools wheel
pip install --no-build-isolation -e '.[test]'
```

For native source changes, follow the [local Gkeyll development workflow](../native/SKILL.md#local-gkeyll-development)
instead of the clean pinned installation commands above. It covers direct
producer edits, in-place builds, and leaving changes uncommitted.

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

Shared skills live in `.agents/skills/`, which Codex discovers directly.
`.claude` is a relative symlink to `.agents/`; root `CLAUDE.md` links to
`AGENTS.md`. Keep `.codex/` a real directory for Codex-specific configuration.
Do not symlink the top-level `.codex` directory: the Linux sandbox cannot
enforce its read-only protection through a writable symlink and fails before
any command runs. Keep `.codex/.gitkeep` so Git retains the directory when
there is no Codex configuration. Put focused skills in
`.agents/skills/<name>/SKILL.md` with name and description frontmatter. Keep
root instructions short and route only to skills relevant to the task.

When present, preserve Entire's Claude hooks in `.agents/settings.json`
(exposed through `.claude/settings.json`) and Codex hooks in `.codex/hooks.json`,
including matchers, commands, and timeouts. Keep one authoritative file for each
agent's hooks; do not copy them between directories.
Leave `.entire/` and Entire's Git hooks intact. Retain existing agent definitions
and historical plans when reorganizing folders; historical markdown remains
ignored, while new skill entry points must be visible to Git.

For instruction/configuration-only changes, validate skill frontmatter, relative
links, Git visibility, and hook preservation; running scientific tests is not
necessary unless application behavior changes.

## Formatting

Run formatting and lint hooks from the repository root after edits. For a focused
check, pass every changed or new file explicitly:

```bash
pre-commit run --files path/to/changed_file.py path/to/changed_file.c
```

For the same full check as CI, run:

```bash
pre-commit validate-config
pre-commit run --all-files --show-diff-on-failure
```

`.pre-commit-config.yaml` pins the tool versions. YAPF formats Python using
`.style.yapf` (two-space indentation, 80 columns); clang-format formats C/C++
using `.clang-format`. Ruff checks lint rules from `pyproject.toml`.
The hooks also check TOML/YAML, merge conflicts, trailing whitespace, and final
newlines. If hooks rewrite files, review the diff and rerun until they pass.
`--all-files` covers tracked files; pass new files with `--files` until Git tracks
them.
