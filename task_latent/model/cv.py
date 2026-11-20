"""
Run-wise split-half cross-validation utilities for multi-task fMRI regression.

Whole runs are held out so that temporally adjacent samples never straddle the
train/validation boundary.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import block_diag

from task_latent.dataset.loader import Dataset

DesignMode = Literal["block", "shared"]


@dataclass(frozen=True)
class Fold:
    """Scan indices (into the Dataset lists) used for training and validation."""

    train: list[int]
    val: list[int]


def _run_sort_key(meta) -> tuple[str, str, int]:
    run = str(meta.run) if meta.run is not None else ""
    return (meta.ped, str(meta.session), int(run) if run.isdigit() else 0)


def make_run_splits(data: Dataset) -> list[Fold]:
    """
    Build two split-half folds (A->B and B->A) by assigning whole runs to halves.

    Runs within a task are sorted by (ped, session, run) and alternated between the
    halves, which keeps AP/PA runs balanced across halves. Tasks with a single run
    are always in the training set and never validated.
    """
    half: dict[int, int] = {}
    single_run: list[int] = []
    for runs in data.task_index.values():
        if len(runs) == 1:
            single_run.extend(runs)
            continue
        # loader order is not deterministic, so sort before assigning halves
        ordered = sorted(runs, key=lambda r: _run_sort_key(data.scan_metadata[r]))
        # alternating over the sorted list puts AP and PA of a 2-run task in opposite halves
        for i, r in enumerate(ordered):
            half[r] = i % 2

    if not half:
        raise ValueError("No task has more than one run; cannot build CV folds.")

    halves = [[r for r, h in half.items() if h == k] for k in (0, 1)]
    # single-run tasks join the training side of both folds
    return [
        Fold(train=sorted(halves[0] + single_run), val=sorted(halves[1])),
        Fold(train=sorted(halves[1] + single_run), val=sorted(halves[0])),
    ]


def task_column_slices(
    task_index: dict[str, list[int]], X_runs: dict[int, np.ndarray]
) -> dict[str, slice]:
    """Column slice of each task's block in the block-diagonal design (task order = task_index)."""
    slices = {}
    start = 0
    for task, runs in task_index.items():
        if not runs:
            continue
        width = X_runs[runs[0]].shape[1]
        slices[task] = slice(start, start + width)
        start += width
    return slices


def build_train_matrices(
    fold: Fold,
    task_index: dict[str, list[int]],
    X_runs: dict[int, np.ndarray],
    Y_runs: list[np.ndarray],
    *,
    design: DesignMode = "block",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a training design and stacked fMRI in task/run order.

    ``block`` assigns each task a separate coefficient block. ``shared``
    vertically stacks runs into one common feature space; its columns must
    already have identical labels and order across all runs. Every task must
    have a training run, as in the original block-diagonal workflow.
    """
    if design not in ("block", "shared"):
        raise ValueError("design must be 'block' or 'shared'")
    train = set(fold.train)
    X_blocks, Y_blocks = [], []
    for runs in task_index.values():
        sel = [r for r in runs if r in train]
        if not sel:
            raise ValueError("A task has no training runs in this fold.")
        # stack a task's training runs in time; each task gets its own column block
        X_blocks.append(np.vstack([X_runs[r] for r in sel]))
        Y_blocks.append(np.vstack([Y_runs[r] for r in sel]))
    # Block order must match task_column_slices so coefficient rows line up.
    if design == "shared":
        widths = {X.shape[1] for X in X_blocks}
        if len(widths) > 1:
            raise ValueError("shared designs must have the same number of columns")
        return np.vstack(X_blocks), np.vstack(Y_blocks)
    return block_diag(*X_blocks), np.vstack(Y_blocks)


def score_fold(
    coef: np.ndarray,
    fold: Fold,
    task_index: dict[str, list[int]],
    X_runs: dict[int, np.ndarray],
    Y_runs: list[np.ndarray],
    col_slices: dict[str, slice] | None = None,
    *,
    design: DesignMode = "block",
) -> dict[str, float]:
    """Score held-out runs using task-specific or shared coefficient rows.

    ``coef`` has shape (features, voxels), including all task blocks in
    ``block`` mode. For scikit-learn multi-output estimators, transpose
    ``coef_`` before passing it. In ``shared`` mode every run uses the same
    rows, and its columns must already have the same labels and order.
    ``col_slices`` identifies task blocks in block mode; if omitted, it is
    derived from ``task_index`` and ``X_runs``. Do not supply it for shared
    mode. Use the same design mode in ``build_train_matrices``.

    Per task, target data are demeaned independently within each held-out run;
    predictions are not demeaned and no intercept is added. SSR and SST are
    pooled across the task's held-out runs, then voxelwise R² is averaged.
    Tasks without validation runs are omitted. An SST floor of 1e-12 is used
    for flat voxels. Fit without an intercept to use this scoring convention.
    """
    if design not in ("block", "shared"):
        raise ValueError("design must be 'block' or 'shared'")
    if coef.ndim != 2:
        raise ValueError("coef must have shape (features, voxels)")
    if design == "shared" and col_slices is not None:
        raise ValueError("col_slices must be omitted for shared designs")
    if design == "block" and col_slices is None:
        col_slices = task_column_slices(task_index, X_runs)
    val = set(fold.val)
    scores = {}
    for task, runs in task_index.items():
        sel = [r for r in runs if r in val]
        if not sel:
            continue
        if design == "shared":
            B = coef
        else:
            assert col_slices is not None
            B = coef[col_slices[task]]
        ssr = 0.0
        sst = 0.0
        for r in sel:
            if X_runs[r].ndim != 2 or X_runs[r].shape[1] != B.shape[0]:
                raise ValueError(
                    f"Run {r}: design width does not match coefficient rows"
                )
            if Y_runs[r].ndim != 2 or Y_runs[r].shape != (
                X_runs[r].shape[0],
                B.shape[1],
            ):
                raise ValueError(
                    f"Run {r}: target shape does not match design and coefficients"
                )
            # demean per run so the baseline is each run's own mean, not the train mean
            Y = Y_runs[r] - Y_runs[r].mean(axis=0)
            ssr = ssr + ((Y - X_runs[r] @ B) ** 2).sum(axis=0)
            sst = sst + (Y**2).sum(axis=0)
        # R2 per voxel, then averaged; the floor avoids dividing by zero for flat voxels
        scores[task] = float(np.mean(1.0 - ssr / np.maximum(sst, 1e-12)))
    return scores
