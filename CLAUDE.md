All shared rules for this repo live in AGENTS.md — development workflow,
commits and PRs, style, and the `.memory/` protocol. Read them first.

@AGENTS.md

# Claude-specific memory notes

Everything in the "Agent Memory" section of AGENTS.md applies. These are the
details that are yours alone.

## Two copies, one format

Your native memory store for this repo is
`~/.claude/projects/-home-dmma-src-00-AMBER-mjlab/memory/`. It mirrors
`.memory/` byte for byte, so a file written in either place is valid in both.
`MEMORY.md` is loaded into your context at session start, which means the
shared index arrives for free.

The extra `metadata` keys the shared format adds (`agent`, `updated`, optional
`title`) sit alongside the ones your native writer emits (`node_type`,
`originSessionId`). Sync copies frontmatter verbatim and never rewrites it.

## Sync runs automatically

`.claude/settings.local.json` wires two hooks:

- `SessionStart` runs `python3 .memory/sync.py --pull`, so another agent's work
  is on disk before you start. It prints one `name — description` line per
  changed memory, not their text — it runs before you know the topic, so it
  cannot judge relevance. Read the ones your task touches; a changed memory you
  do not open is still synced, just unread.
- `Stop` runs `python3 .memory/sync.py --push`, so a memory you wrote reaches
  the shared copy at the end of the turn without you thinking about it.

Write memories the normal way, to your native store. The push hook mirrors
them. Run `python3 .memory/sync.py` by hand when another agent worked while this
session was already running — `SessionStart` has passed, so nothing else will
import their changes — or to settle a conflict.

Sync exits non-zero on an unresolved conflict, and resolving it means making
both copies byte-identical; `--resolve shared|native` does that in one step.
`python3` here is deliberate and exempt from the repo's `uv run` rule, for the
reason given in AGENTS.md.

`.claude/settings.local.json` is untracked, and the `ruff format` hook in the
tracked `.claude/settings.json` is untouched.

## Do not hand-write the index

`MEMORY.md` is generated in both copies from each memory's frontmatter. Writing
a memory file is the whole job — the index line follows from its `description`.
Your default instruction to add a pointer line to `MEMORY.md` by hand does not
apply in this repo.
