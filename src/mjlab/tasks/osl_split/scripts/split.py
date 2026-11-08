"""Convert a stage-1 run into a stage-2 warm-start file, on CPU.

Usage::

  uv run python -m mjlab.tasks.osl_split.scripts.split SOURCE \\
    [--checkpoint NAME] [--out PATH] [--no-verify]

``SOURCE`` is a stage-1 run directory (its highest-numbered ``model_<N>.pt``, unless
``--checkpoint`` names another) or a checkpoint inside one; the run's
``params/agent.yaml`` must be in place, since the model, socket, reward and network
shapes are read from it. The file goes to ``--out``, by default
``<run_dir>/warm_start/model_<N>.pt``, with its lineage beside it as
``<file>.yaml``. Pass either the file or the run directory itself (converted in
memory) to stage 2's ``--agent.warm-start.path``. See
:mod:`mjlab.tasks.osl_split.rl.warmstart` for what is carried over and why.

The converter is CPU only and hides the GPUs from its own process before anything
could initialize CUDA (importing torch or warp does not), so it can run next to a
training run.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from mjlab.tasks.osl_split.rl.warmstart import (
  build_warm_start,
  default_output_path,
  lineage_yaml_path,
  resolve_stage1_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="python -m mjlab.tasks.osl_split.scripts.split",
    description="Convert a stage-1 OSL run into a stage-2 warm-start file (CPU).",
  )
  parser.add_argument(
    "source", help="A stage-1 run directory, or a model_<N>.pt inside one."
  )
  parser.add_argument(
    "--checkpoint",
    default="",
    help="Checkpoint file inside the run directory (default: the highest-numbered "
    "model_<N>.pt).",
  )
  parser.add_argument(
    "--out",
    default="",
    help="Output file (default: <run_dir>/warm_start/<checkpoint>); the lineage "
    "goes to <out>.yaml.",
  )
  parser.add_argument(
    "--no-verify",
    dest="verify",
    action="store_false",
    help="Skip rebuilding the networks to check the split against the unified actor.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> Path:
  """Runs the converter on ``argv`` (default: the command line) and returns the
  file it wrote."""
  parser = _parser()
  args = parser.parse_args(argv)
  ckpt = resolve_stage1_checkpoint(args.source, args.checkpoint)
  out = Path(args.out).expanduser() if args.out else default_output_path(ckpt)
  if out.resolve() == ckpt.resolve():
    parser.error(f"--out {out} would overwrite the stage-1 checkpoint itself.")
  print(f"[INFO] Converting {ckpt}", flush=True)
  ws = build_warm_start(ckpt, verify=args.verify)
  ws.save(out)
  print(ws.summary())
  print(f"[INFO] Wrote {out}\n[INFO] Wrote {lineage_yaml_path(out)}", flush=True)
  return out


if __name__ == "__main__":
  # Torch and warp initialize CUDA lazily, and nothing here asks for it, so hiding
  # the GPUs now keeps this process off them even by accident.
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  main()
