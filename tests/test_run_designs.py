import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from main import build_run_designs
from task_latent.dataset.loader import Dataset, ScanMetadata
from task_latent.model.basis import HRFBasis, convolve_event_with_basis


def events(*labels):
    return pd.DataFrame(
        {
            "trial_type": list(labels),
            "onset": [float(i * 3) for i in range(len(labels))],
            "duration": [1.0] * len(labels),
            "amplitude": [1.0] * len(labels),
        }
    )


def dataset(event_runs, task_index):
    metadata = []
    for run_idx, event_df in enumerate(event_runs):
        task = next(task for task, runs in task_index.items() if run_idx in runs)
        metadata.append(
            ScanMetadata(
                label=task,
                conditions=event_df["trial_type"].dropna().unique().tolist(),
                tr=1.0,
                n_frames=15 + run_idx,
                ped="ap",
                session="01",
                run=str(run_idx),
            )
        )
    return Dataset(
        fmri=[],
        events=event_runs,
        scan_metadata=metadata,
        task_index=task_index,
        task_conditions={},
        subject="test",
        normalize=False,
    )


class RunDesignTests(unittest.TestCase):
    def setUp(self):
        self.basis = HRFBasis(duration_sec=8, knots_per_sec=0.5).create(dt=0.1)

    def test_global_alignment_preserves_convolved_values_and_zero_fills(self):
        data = dataset(
            [
                events("visual-left", "shared"),
                events("shared", "auditory-right"),
                events("auditory-right", "task-b-only"),
                events(None),
            ],
            {"A": [0, 1], "B": [2, 3]},
        )
        designs = build_run_designs(data, self.basis)
        labels = designs[0][1]
        self.assertEqual(labels, sorted(labels))
        self.assertEqual(len(labels), 4 * self.basis.basis.shape[1])
        for idx in range(3):
            X, run_labels = designs[idx]
            self.assertEqual(run_labels, labels)
            original, local_labels = convolve_event_with_basis(
                data.events[idx],
                data.scan_metadata[idx].tr,
                data.scan_metadata[idx].n_frames,
                self.basis,
            )
            positions = [labels.index(label) for label in local_labels]
            assert_allclose(X[:, positions], original)
            missing = [i for i, label in enumerate(labels) if label not in local_labels]
            assert_array_equal(X[:, missing], 0)
        self.assertEqual(designs[3][1], labels)
        assert_array_equal(designs[3][0], 0)
        self.assertEqual(designs[3][0].shape, (18, len(labels)))
        self.assertIsNot(designs[0][1], designs[1][1])

    def test_empty_events_and_empty_dataset(self):
        data = dataset([events(), events(None)], {"A": [0, 1]})
        designs = build_run_designs(data, self.basis)
        self.assertEqual(designs[0][0].shape, (15, 0))
        self.assertEqual(designs[1][0].shape, (16, 0))
        self.assertEqual(designs[0][1], [])
        self.assertEqual(build_run_designs(dataset([], {}), self.basis), {})

    def test_unmapped_rows_are_omitted_from_nonempty_run(self):
        data = dataset([events(None, "shared")], {"A": [0]})
        design, labels = build_run_designs(data, self.basis)[0]
        expected, expected_labels = convolve_event_with_basis(
            data.events[0].dropna(subset=["trial_type"]), 1.0, 15, self.basis
        )
        self.assertEqual(labels, expected_labels)
        assert_allclose(design, expected)

    def test_reordered_labels_and_invalid_convolution_output(self):
        data = dataset([events("a"), events("b")], {"A": [0, 1]})
        outputs = [
            (np.tile([1.0, 2.0], (15, 1)), ["z", "a"]),
            (np.tile([3.0, 4.0], (16, 1)), ["a", "z"]),
        ]
        with patch("main.convolve_event_with_basis", side_effect=outputs):
            result = build_run_designs(data, self.basis)
        self.assertEqual(result[0][1], ["a", "z"])
        assert_array_equal(result[0][0][0], [2.0, 1.0])
        assert_array_equal(result[1][0][0], [3.0, 4.0])

        with (
            patch(
                "main.convolve_event_with_basis",
                return_value=(np.ones((15, 2)), ["a", "a"]),
            ),
            self.assertRaisesRegex(ValueError, "duplicate design labels"),
        ):
            build_run_designs(data, self.basis)
        with (
            patch(
                "main.convolve_event_with_basis", return_value=(np.ones((15, 2)), ["a"])
            ),
            self.assertRaisesRegex(ValueError, "design width does not match labels"),
        ):
            build_run_designs(data, self.basis)


if __name__ == "__main__":
    unittest.main()
