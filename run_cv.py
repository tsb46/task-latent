"""
Perform cross-validation on the IBC dataset using shared-feature ridge regression.
"""

import argparse
import itertools
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from main import ALLOWED_NAMESPACE_CROSS_PAIRS, ANNOTATION_NAMESPACES, build_run_designs
from task_latent.dataset.loader import DataLoader
from task_latent.model.basis import HRFBasis
from task_latent.model.cv import (
    build_train_matrices,
    make_run_splits,
    score_fold,
)
from task_latent.tagging.annotations import (
    AnnotationMapper,
    load_annotations,
)

# List of tasks to be used in the cross-validation
TASKS = [
    "ArchiStandard",
    "ArchiSpatial",
    "ArchiSocial",
    "ArchiEmotional",
    "HcpEmotion",
    "HcpGambling",
    "HcpMotor",
    "HcpLanguage",
    "HcpRelational",
    "HcpSocial",
    "HcpWm",
    "RSVPLanguage",
    # "MTTWE",
    # "MTTNS",
    # "PreferenceFood",
    # "PreferencePaintings",
    # "PreferenceFaces",
    # "PreferenceHouses",
    # "TheoryOfMind",
    # "EmotionalPain",
    # "PainMovie",
    # "VSTM",
    # "Enumeration",
    # "Self",
    # "Bang",
    # "WedgeClock",
    # "WedgeAnti",
    # "ContRing",
    # "ExpRing",
    # "Lec2",
    # "Audi",
    # "Visu",
    # "Lec1",
    # "MVEB",
    # "MVIS",
    # "Moto",
    # "MCSE",
    # "Audio",
    # "Attention",
    # "StopSignal",
    # "TwoByTwo",
    # "SelectiveStopSignal",
    # "Stroop",
    # "ColumbiaCards",
    # "DotPatterns",
    # "WardAndAllport",
    # "BiologicalMotion1",
    # "BiologicalMotion2",
    # "MathLanguage",
    # "SpatialNavigation",
    # "EmoMem",
    # "EmoReco",
    # "StopNogo",
    # "Catell",
    # "FingerTapping",
    # "VSTMC",
    # "RewProc",
    # "NARPS",
    # "FaceBody",
    # "Scene",
    # "BreathHolding",
    # "Checkerboard",
    # "FingerTap",
    # "ItemRecognition",
    # "VisualSearch",
    # "Color",
    # "Motion",
]

# Parameter search space for cross-validation
PARAM_GRID = {
    # "lambda_nuclear": [10.0, 50.0, 100.0, 500.0, 1000.0],
    "lambda_l2": [0.001, 0.01, 0.1, 1.0, 10.0],
}


def run_cross_validation(
    dataset_name: Literal["ibc"],
    subject_id: str,
    out_path: str | None = None,
) -> None:
    """
    Tune ridge regularization with run-wise split-half CV on shared features.

    Runs within each task are split into two balanced halves (see `make_run_splits`);
    the model is trained on one half of every task and scored on the other, then the
    roles are swapped. Tasks with a single run are only used for training.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use (currently only "ibc" is supported).
    subject_id : str
        Subject ID to use for the cross-validation (e.g., "04").
    out_path : str or None, optional
        If given, long-format results (one row per parameters/fold/task) are saved here as CSV.

    Returns
    -------
    None
        Results are printed and optionally saved to CSV.
    """
    loader = DataLoader(dataset_name, subject_id)

    # Load annotations for mapping event annotations to features
    annotations = load_annotations()
    mapper = AnnotationMapper(
        annotations,
        namespaces=ANNOTATION_NAMESPACES,
        exclude_none=ANNOTATION_NAMESPACES,
    )
    annot_mapper = mapper.map_combinations(
        order=2, namespace_pairs=ALLOWED_NAMESPACE_CROSS_PAIRS
    )
    data = loader.load(
        tasks=TASKS,
        # annotation_mapping=annot_mapper
    )

    hrf_basis = HRFBasis(
        duration_sec=30.0,
        knot_spacing="uniform",
        basis_type="cr",
        zero_at_onset=True,
    )
    hrf_basis.create()

    # the design does not depend on hyperparameters, so build it once
    X_runs = build_run_designs(data, hrf_basis)
    # two folds: train on half of each task's runs, validate on the other half, then swap
    folds = make_run_splits(data)
    # training matrices depend only on the fold, so build them once outside the grid loop
    fold_matrices = [
        build_train_matrices(f, data.task_index, X_runs, data.fmri, design="block")
        for f in folds
    ]
    # single-run tasks are train-only, so they do not appear in the scored list
    for i, f in enumerate(folds):
        scored = sorted({data.scan_metadata[r].label for r in f.val})
        print(
            f"Fold {i}: {len(f.train)} train runs, {len(f.val)} val runs, "
            f"scored tasks: {scored}"
        )

    param_grid = PARAM_GRID
    rows = []
    grid = list(itertools.product(*param_grid.values()))
    for n, values in enumerate(grid, start=1):
        params = dict(zip(param_grid.keys(), values))
        print(f"[{n}/{len(grid)}] {params}")
        for fold_idx, (fold, (X_train, Y_train)) in enumerate(
            zip(folds, fold_matrices)
        ):
            # score_fold evaluates predictions without an intercept against
            # within-run demeaned validation targets.
            model = Ridge(alpha=params["lambda_l2"], fit_intercept=False)
            model.fit(X_train, Y_train)
            # sklearn stores coefficients as (voxels, features); scoring uses
            # (features, voxels), shared across all tasks.
            scores = score_fold(
                model.coef_.T, fold, data.task_index, X_runs, data.fmri, design="shared"
            )
            for task, r2 in scores.items():
                rows.append(
                    {
                        **params,
                        "fold": fold_idx,
                        "task": task,
                        "r2": r2,
                    }
                )
            print(f"  fold {fold_idx}: mean R2 = {np.mean(list(scores.values())):.4f}")

    results = pd.DataFrame(rows)
    if out_path:
        results.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")

    # mean over tasks within a fold first so tasks with more val runs are not overweighted
    keys = list(param_grid.keys())
    summary = (
        results.groupby([*keys, "fold"], as_index=False)["r2"]
        .mean()
        .groupby(keys, as_index=False)["r2"]
        .mean()
        .sort_values("r2", ascending=False)
    )
    print("\nMean validation R2 by hyperparameters:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run cross-validation on the IBC dataset using shared-feature ridge regression."
    )
    parser.add_argument("-s", "--subject", type=str, help="Subject ID (e.g., '04')")
    parser.add_argument(
        "-d", "--dataset", type=str, default="ibc", help="Dataset name (default: ibc)"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="cv_results.csv",
        help="CSV path for CV results",
    )
    args = parser.parse_args()

    run_cross_validation(
        dataset_name=args.dataset,
        subject_id=args.subject,
        out_path=args.out,
    )
