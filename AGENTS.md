# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# Agent Memory

`.memory/` is shared working memory for every agent on this repo — Claude Code,
Codex, or anything else. It exists so that the agent picking up a thread does
not have to re-derive what the previous one already worked out.

## Start of session

**Read `.memory/MEMORY.md`, then open the memories your task touches.** That is
mandatory, and sync does not do it for you. Sync reports whether two folders
agree; it says nothing about what this session has read. "All in sync" on a cold
session means you have read nothing.

Claude Code keeps a second, native copy of these memories and imports changes
through a `SessionStart` hook. Other agents have nothing to import — `.memory/`
is their only store — so they need no sync at startup at all.

## Commands

**The `uv run` rule above does not apply to `.memory/sync.py`.** It is a
dependency-free stdlib script that has to keep working when the project
environment is broken, half-synced or absent, which is exactly when memory
matters most. Call it with `python3`.

| you are | at startup | after writing or editing a memory |
| --- | --- | --- |
| Claude Code | nothing — the `SessionStart` hook pulls | nothing — the `Stop` hook pushes |
| any other agent | read `MEMORY.md` | `python3 .memory/sync.py --repo-only` |

`--repo-only` rebuilds `.memory/MEMORY.md` and writes nothing outside the
working tree, so it needs no permissions beyond the repo. Claude imports those
changes at its next session start.

Sync reports changed memories as one `name — description` line each, never as
full text: it runs before a session has a topic, so it cannot judge relevance.
The description is the hook — open the files your task actually touches.
`--print-bodies` dumps full text when you genuinely want all of it.
`--resolve` cannot be combined with `--pull`/`--push`; it moves whichever side
it is told to keep, which would contradict the direction those flags promise.

## What is worth writing

Write a memory when a session produced something the next agent would
otherwise rediscover the hard way:

- a change and the reasoning behind it, especially a decision that could
  plausibly have gone the other way
- a problem hit and how it was resolved, including wrong turns worth not
  repeating
- behavior of this codebase that is not visible from the code you would
  naturally read
- a correction to something an earlier memory got wrong

Do not write what the repo already records: file layout, what a function does,
anything recoverable from `git log`.

Write for the actual reader: a different agent, starting cold, mid-thread, with
no transcript. *Why* beats *what* — the diff is already readable. Say plainly
what is done, what is in progress, and what is untested.

## Format

One file per topic, `kebab-case.md`, directly in `.memory/`:

```markdown
---
name: dof-ordering-gotcha
description: "One line. This is what lands in MEMORY.md, so make it the hook."
metadata:
  type: project
  agent: claude
  updated: 2026-09-16
---

Body. Bold the load-bearing gotcha. Separate what was verified from what was
assumed. End with **How to apply:**. Link related memories as
[[other-memory]] — a link to a memory that does not exist yet is fine, it
marks something worth writing.
```

`type` is one of:

| type | use for |
| --- | --- |
| `project` | the work itself: what was built, why, where it stands, what is open. The default. |
| `reference` | durable facts that hold regardless of the current goal — codebase mechanics, dataset locations, external sources. |
| `feedback` | a rule the user gave about how to work. Include why, and how to apply it. |
| `user` | something about the user that changes how work should be done here. |

Optional `title:` overrides the index title derived from `name`.

## Rules

- **One topic per file, amended in place.** When new work touches an existing
  memory, edit that file and bump `updated` rather than adding a near-duplicate.
  Correct wrong claims in place; do not leave them standing next to the fix.
- **Set `agent:` to yourself** when you write or amend a file, so a claim can be
  traced back.
- **Keep files under ~150 lines.** Past that, split by sub-topic and link the
  pieces.
- **Never hand-edit `.memory/MEMORY.md`.** It is regenerated from every file's
  `name`, `title`, `description` and `type`. To change an index line, change the
  memory's `description`.
- **Regenerate the index after writing**, per the command table above.

## Conflicts

Sync compares the shared copy, the native copy and the hashes recorded at the
last sync. A file changed on one side is copied. A file changed on **both** is a
conflict: neither copy is overwritten, a unified diff of the two is printed, and
sync exits non-zero.

**Resolving means making both copies byte-identical.** Editing only the side you
happen to be looking at leaves the conflict standing, because on the next run
neither copy will match the recorded hash. Write the merged text to both, or run
`python3 .memory/sync.py --resolve shared|native` to keep one side wholesale.

Conflicts are rare here — the usual pattern is one agent handing off to another,
not two writing at once.

## Git

`.memory/` is deliberately untracked in this repo (listed in
`.git/info/exclude`), because `origin` is the public `mujocolab/mjlab` and
working memory does not belong in upstream history.

`AGENTS.md` and `CLAUDE.md` are tracked upstream, and the memory sections in
them are local modifications that are **never committed**. In this repo:

- never `git add AGENTS.md` or `git add CLAUDE.md`
- never `git commit -a` — always name the paths being committed
