"""
Run-wise split-half cross-validation utilities for multi-task fMRI regression.

Whole runs are held out so that temporally adjacent samples never straddle the
train/validation boundary.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from task_latent.dataset.loader import Dataset


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
) -> tuple[np.ndarray, np.ndarray]:
    """Block-diagonal training design (tasks on the diagonal) and stacked fMRI."""
    train = set(fold.train)
    X_blocks, Y_blocks = [], []
    for runs in task_index.values():
        sel = [r for r in runs if r in train]
        if not sel:
            raise ValueError("A task has no training runs in this fold.")
        # stack a task's training runs in time; each task gets its own column block
        X_blocks.append(np.vstack([X_runs[r] for r in sel]))
        Y_blocks.append(np.vstack([Y_runs[r] for r in sel]))
    # task order here must match task_column_slices so coefficient rows line up
    return block_diag(*X_blocks), np.vstack(Y_blocks)


def score_fold(
    coef: np.ndarray,
    fold: Fold,
    task_index: dict[str, list[int]],
    X_runs: dict[int, np.ndarray],
    Y_runs: list[np.ndarray],
    col_slices: dict[str, slice],
) -> dict[str, float]:
    """
    Per-task validation R2: voxelwise 1 - SSR/SST pooled over the task's held-out
    runs (each run demeaned), averaged over voxels.
    """
    val = set(fold.val)
    scores = {}
    for task, runs in task_index.items():
        sel = [r for r in runs if r in val]
        if not sel:
            continue
        # the design is block-diagonal, so a run is predicted from its own task's rows only
        B = coef[col_slices[task]]
        ssr = 0.0
        sst = 0.0
        for r in sel:
            # demean per run so the baseline is each run's own mean, not the train mean
            Y = Y_runs[r] - Y_runs[r].mean(axis=0)
            ssr = ssr + ((Y - X_runs[r] @ B) ** 2).sum(axis=0)
            sst = sst + (Y**2).sum(axis=0)
        # R2 per voxel, then averaged; the floor avoids dividing by zero for flat voxels
        scores[task] = float(np.mean(1.0 - ssr / np.maximum(sst, 1e-12)))
    return scores
