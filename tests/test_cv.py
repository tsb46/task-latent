import unittest
from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Ridge

from task_latent.model.cv import (
    Fold,
    build_train_matrices,
    score_fold,
    task_column_slices,
)


class CrossValidationDesignTests(unittest.TestCase):
    def setUp(self):
        # A and B share both feature columns; C is a train-only task.
        self.tasks = {"A": [0, 1, 2], "B": [3, 4], "C": [5]}
        self.X = {
            0: np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            1: np.array([[2.0, 0.0], [0.0, 1.0], [-2.0, -1.0]]),
            2: np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]]),
            3: np.array([[1.0, 1.0], [1.0, -1.0], [-2.0, 0.0]]),
            4: np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]),
            5: np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        }
        self.B = np.array([[2.0, -1.0], [1.0, 3.0]])
        self.Y = [self.X[i] @ self.B + (i + 1) * 5 for i in range(6)]
        self.fold = Fold(train=[0, 3, 5], val=[1, 2, 4])

    def test_shared_design_stack_and_fit_predict(self):
        X_train, Y_train = build_train_matrices(
            self.fold, self.tasks, self.X, self.Y, design="shared"
        )
        assert_array_equal(X_train, np.vstack([self.X[i] for i in (0, 3, 5)]))
        assert_array_equal(Y_train, np.vstack([self.Y[i] for i in (0, 3, 5)]))
        scores = score_fold(
            self.B, self.fold, self.tasks, self.X, self.Y, design="shared"
        )
        self.assertEqual(set(scores), {"A", "B"})
        assert_allclose(list(scores.values()), [1.0, 1.0])

        # Real sklearn coefficients are voxels x features and must be transposed.
        model = Ridge(alpha=0.1, fit_intercept=False).fit(X_train, Y_train)
        actual = score_fold(
            model.coef_.T, self.fold, self.tasks, self.X, self.Y, design="shared"
        )
        for task, runs in (("A", (1, 2)), ("B", (4,))):
            residual = np.zeros(2)
            total = np.zeros(2)
            for run in runs:
                target = self.Y[run] - self.Y[run].mean(axis=0)
                residual += ((target - model.predict(self.X[run])) ** 2).sum(axis=0)
                total += (target**2).sum(axis=0)
            self.assertAlmostEqual(actual[task], np.mean(1 - residual / total))

    def test_block_design_uses_task_specific_rows(self):
        X_train, Y_train = build_train_matrices(self.fold, self.tasks, self.X, self.Y)
        self.assertEqual(X_train.shape, (9, 6))
        assert_array_equal(X_train[:3, :2], self.X[0])
        assert_array_equal(X_train[:3, 2:], 0)
        assert_array_equal(X_train[3:6, 2:4], self.X[3])
        assert_array_equal(Y_train, np.vstack([self.Y[i] for i in (0, 3, 5)]))
        slices = task_column_slices(self.tasks, self.X)
        coef = np.vstack([self.B, self.B * 0, self.B])
        expected = {"A": 1.0, "B": 0.0}
        scores = score_fold(coef, self.fold, self.tasks, self.X, self.Y, slices)
        for task, value in expected.items():
            self.assertAlmostEqual(scores[task], value)
        self.assertEqual(
            scores, score_fold(coef, self.fold, self.tasks, self.X, self.Y)
        )

        model = Ridge(alpha=1.0, fit_intercept=False).fit(X_train, Y_train)
        actual = score_fold(
            model.coef_.T, self.fold, self.tasks, self.X, self.Y, slices
        )
        for task, runs in (("A", (1, 2)), ("B", (4,))):
            ssr, sst = np.zeros(2), np.zeros(2)
            for run in runs:
                target = self.Y[run] - self.Y[run].mean(axis=0)
                full = np.zeros((len(target), X_train.shape[1]))
                full[:, slices[task]] = self.X[run]
                ssr += ((target - model.predict(full)) ** 2).sum(axis=0)
                sst += (target**2).sum(axis=0)
            self.assertAlmostEqual(actual[task], np.mean(1 - ssr / sst))

    def test_dimension_and_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "design must"):
            score_fold(
                self.B, self.fold, self.tasks, self.X, self.Y, design=cast(Any, "bad")
            )
        with self.assertRaisesRegex(ValueError, "design must"):
            build_train_matrices(
                self.fold, self.tasks, self.X, self.Y, design=cast(Any, "bad")
            )
        with self.assertRaisesRegex(ValueError, "col_slices must be omitted"):
            score_fold(
                self.B, self.fold, self.tasks, self.X, self.Y, {}, design="shared"
            )
        with self.assertRaisesRegex(ValueError, "coef must have shape"):
            score_fold(
                self.B.ravel(), self.fold, self.tasks, self.X, self.Y, design="shared"
            )
        with self.assertRaisesRegex(ValueError, "design width"):
            score_fold(
                self.B[:1], self.fold, self.tasks, self.X, self.Y, design="shared"
            )
        with self.assertRaisesRegex(ValueError, "target shape"):
            score_fold(
                self.B[:, :1], self.fold, self.tasks, self.X, self.Y, design="shared"
            )
        incompatible = dict(self.X)
        incompatible[3] = incompatible[3][:, :1]
        with self.assertRaisesRegex(ValueError, "same number of columns"):
            build_train_matrices(
                self.fold, self.tasks, incompatible, self.Y, design="shared"
            )


if __name__ == "__main__":
    unittest.main()
