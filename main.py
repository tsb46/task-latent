"""
To-be-implemented main script for fitting a regression on the IBC dataset and saving R2 and brain maps.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from task_latent.dataset.loader import DataLoader, Dataset
from task_latent.model.basis import HRFBasis, convolve_event_with_basis
from task_latent.tagging.annotations import (
    AnnotationMapper,
    load_annotations,
)

# List of tasks to be used in analysis
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
    "MTTWE",
    "MTTNS",
    "PreferenceFood",
    "PreferencePaintings",
    "PreferenceFaces",
    "PreferenceHouses",
    "TheoryOfMind",
    "EmotionalPain",
    "PainMovie",
    "VSTM",
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

# Annotation namespaces to be used for mapping event annotations to features
ANNOTATION_NAMESPACES = [
    "stimulus_modality",
    "stimulus_laterality",
    "stimulus_dynamics",
    "response_effector",
    "response_laterality",
]
# Allowed namespace cross-pairs for annotation mapping
ALLOWED_NAMESPACE_CROSS_PAIRS = [
    ("stimulus_modality", "stimulus_laterality"),
    ("stimulus_modality", "stimulus_dynamics"),
    ("response_effector", "response_laterality"),
]


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / f"sub-{args.subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations for mapping event annotations to features
    annotations = load_annotations()
    mapper = AnnotationMapper(
        annotations,
        namespaces=ANNOTATION_NAMESPACES,
        exclude_none=ANNOTATION_NAMESPACES,
    )
    annot_mapper, feature_counts = mapper.map_combinations(
        order=2, return_counts=True, namespace_pairs=ALLOWED_NAMESPACE_CROSS_PAIRS
    )

    if args.verbose > 0:
        print("Loading dataset for subject ", args.subject)
    loader = DataLoader(args.dataset, args.subject)
    data = loader.load(
        tasks=TASKS,
        # annotation_mapping=annot_mapper,
        verbose=args.verbose,
    )

    if args.verbose > 0:
        print("Projecting events onto shared HRF basis...")
    # create HRF basis for modeling
    hrf_basis = HRFBasis(
        duration_sec=30.0,
        knot_spacing="uniform",
        basis_type="cr",
        zero_at_onset=True,
    )
    hrf_basis.create()

    # build run designs
    X_runs = build_run_designs(data, hrf_basis)

    # Stack run designs into a single design matrix
    X = np.vstack([X_run for X_run in X_runs.values()])
    Y = np.vstack(data.fmri)
    if args.verbose > 0:
        print("Stacked design matrix shape: ", X.shape)
        print("Stacked fMRI data shape: ", Y.shape)
        print("Number of runs: ", len(X_runs))

    # Fit a ridge regression model
    if args.verbose > 0:
        print("Fitting ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X, Y)


def build_run_designs(data: Dataset, hrf_basis: HRFBasis) -> dict[int, np.ndarray]:
    """Convolve each run and align its columns to the union across all runs.

    The returned labels are sorted identically for every run. Features absent
    from a run are zero-filled, including runs with no mapped events. Events
    with null trial types have no annotation feature and are omitted.
    """
    X_runs: dict[int, np.ndarray] = {}
    labels_all = []
    for task, runs in data.task_index.items():
        print(f"Projecting task {task} to shared HRF basis...")
        for run_idx in runs:
            meta = data.scan_metadata[run_idx]
            events = data.events[run_idx].dropna(subset=["trial_type"])
            if events.empty:
                X_run, labels = np.zeros((meta.n_frames, 0)), []
            else:
                X_run, labels = convolve_event_with_basis(
                    event_df=events,
                    tr=meta.tr,
                    basis=hrf_basis,
                    n_frames=meta.n_frames,
                )
            X_runs[run_idx] = X_run
            labels_all.append(labels)

    # Label identity, not run-local column position, determines the shared
    # coefficient for a spline regressor across runs and tasks.
    shared_labels = sorted({label for labels in labels_all for label in labels})
    positions = {label: i for i, label in enumerate(shared_labels)}
    for run_idx, X_run in X_runs.items():
        if X_run.ndim != 2 or X_run.shape[1] != len(labels_all[run_idx]):
            raise ValueError(f"Run {run_idx}: design width does not match labels")
        if len(set(labels_all[run_idx])) != len(labels_all[run_idx]):
            raise ValueError(f"Run {run_idx}: duplicate design labels")
        aligned = np.zeros((X_run.shape[0], len(shared_labels)), dtype=X_run.dtype)
        aligned[:, [positions[label] for label in labels_all[run_idx]]] = X_run
        X_runs[run_idx] = aligned
    return X_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a regression on the IBC dataset and save R2 and brain maps."
    )
    parser.add_argument(
        "-s", "--subject", type=str, required=True, help="Subject ID (e.g., '04')"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="ibc", help="Dataset name (default: ibc)"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, help="Verbosity level (default: 1)"
    )

    main(parser.parse_args())
