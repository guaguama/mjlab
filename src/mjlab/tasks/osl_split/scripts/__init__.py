"""Command-line entry points: the warm-start converter and the two-stage pipeline.

Run the converter as a module, ``uv run python -m
mjlab.tasks.osl_split.scripts.split``, and the pipeline driver as a file, ``uv run
--no-sync python src/mjlab/tasks/osl_split/scripts/two_stage.py``: as a module, the
driver would import the mjlab package first, and with it torch, warp and every task,
into a process that only waits on its steps.
"""
