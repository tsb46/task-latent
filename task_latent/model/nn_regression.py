"""
This module implements nuclear norm (low-rank) regularization for high-dimensional
multi-output regression using proximal gradient descent with SVD soft-thresholding.
"""

from typing import Literal

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd


class NuclearNormRegressor:
    """
    Nuclear norm regularized regression with optional rank constraint.

    Solves the optimization problem:
        minimize (1/2)||Y - XB||²_F + (λ₂/2)||B||²_F + λ₁||B||_*

    where:
        - ||·||_F is the Frobenius norm
        - ||·||_* is the nuclear norm (sum of singular values)
        - B is the coefficient matrix (shape P × V)

    The nuclear norm penalty encourages low-rank solutions by shrinking singular
    values. An optional hard rank constraint can be imposed for computational
    efficiency and interpretability.

    Parameters
    ----------
    lambda_nuclear : float
        Nuclear norm regularization parameter (λ₁). Controls rank/singular value
        shrinkage. Larger values produce lower-rank solutions. A practical default
        is 1e-3, but it should be tuned via cross-validation.

    lambda_l2 : float
        L2 (Frobenius) regularization parameter (λ₂). Provides numerical stability
        and additional coefficient shrinkage. A practical default is 10, but it should
        be tuned via cross-validation.

    rank_k : int or None, default=None
        Hard rank constraint. If specified, forces B to have rank exactly K by
        truncating singular values beyond K. Enables use of randomized SVD for
        computational speedup when K << min(P, V). If None, uses pure nuclear
        norm regularization without hard truncation.

    max_iter : int, default=1000
        Maximum number of proximal gradient iterations.

    tol : float, default=1e-4
        Convergence tolerance. Algorithm stops when relative change in B falls
        below this threshold: ||B_new - B_old||_F / ||B_old||_F < tol.

    verbose : int, default=0
        Verbosity level. If greater than 0, prints the current iteration index. If greater than
        1, prints convergence information including objective values and singular values during optimization.

    Attributes
    ----------
    coef_ : ndarray of shape (P, V)
        Fitted coefficient matrix.

    U_ : ndarray of shape (P, K)
        Left singular vectors from final SVD decomposition.

    s_ : ndarray of shape (K,)
        Singular values from final SVD decomposition (soft-thresholded).

    Vt_ : ndarray of shape (K, V)
        Right singular vectors (transposed) from final SVD decomposition.

    n_iter_ : int
        Number of iterations performed.

    loss_history_ : list of float
        Objective function values at each iteration.

    singular_value_history_ : list of ndarray
        Singular values at each iteration (if verbose=True).

    Examples
    --------
    >>> from task_latent.model.nn_regression import NuclearNormRegressor
    >>> import numpy as np
    >>>
    >>> # Generate synthetic low-rank data
    >>> X = np.random.randn(200, 50)
    >>> B_true = np.random.randn(50, 10) @ np.random.randn(10, 100)  # rank 10
    >>> Y = X @ B_true + 0.1 * np.random.randn(200, 100)
    >>>
    >>> # Fit with rank constraint
    >>> model = NuclearNormRegressor(
    ...     lambda_nuclear=1.0,
    ...     lambda_l2=0.1,
    ...     rank_k=10,
    ...     verbose=True
    ... )
    >>> model.fit(X, Y)
    >>> Y_pred = model.predict(X)
    >>> print(f"R² score: {model.score(X, Y):.3f}")
    >>> print(f"Effective rank: {model.get_effective_rank():.1f}")

    References
    ----------
    - Parikh & Boyd (2014). "Proximal Algorithms". Foundations and Trends in Optimization.
    - Cai et al. (2010). "A singular value thresholding algorithm for matrix completion"
    """

    def __init__(
        self,
        lambda_nuclear: float = 1e3,
        lambda_l2: float = 10,
        rank_k: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: int = 0,
    ):
        self.lambda_nuclear = lambda_nuclear
        self.lambda_l2 = lambda_l2
        self.rank_k = rank_k
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NuclearNormRegressor":
        """
        Fit the nuclear norm regression model using proximal gradient descent.

        Parameters
        ----------
        X : ndarray of shape (T, P)
            Design matrix (predictors).

        Y : ndarray of shape (T, V)
            Response matrix (targets).

        Returns
        -------
        self : NuclearNormRegressor
            Fitted estimator.
        """
        # Input validation
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-dimensional arrays")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of samples. "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}"
            )

        T, P = X.shape
        _, V = Y.shape

        if self.rank_k is not None and self.rank_k > min(P, V):
            raise ValueError(
                f"rank_k ({self.rank_k}) cannot exceed min(P, V) = {min(P, V)}"
            )

        # Compute Lipschitz constant and step size
        L = np.linalg.norm(X, ord=2) ** 2 + self.lambda_l2
        step_size = 1.0 / L

        # Precompute the Gram terms once so the inner loop avoids Y - X @ B.
        XtX = X.T @ X
        Xty = X.T @ Y
        XtX_reg = XtX + self.lambda_l2 * np.eye(P, dtype=X.dtype)

        # Initialize coefficient matrix with scaled least squares
        B = Xty / L

        # Track convergence
        self.loss_history_ = []
        if self.verbose > 0:
            self.singular_value_history_ = []
            print("Starting proximal gradient descent...")
            print(f"  Data: T={T}, P={P}, V={V}")
            print(f"  Lipschitz constant L: {L:.4f}")
            print(f"  Step size: {step_size:.6f}")
            print(
                f"  Rank constraint: {self.rank_k if self.rank_k else 'None (pure nuclear norm)'}"
            )
            print()

        # Proximal gradient descent
        # Initialize variables in case max_iter = 0
        iteration = 0
        B_new = B
        U = np.zeros((P, 1))
        s_new = np.zeros(1)
        Vt = np.zeros((1, V))
        objective = 0.0

        for iteration in range(self.max_iter):
            # Compute gradient of smooth part
            gradient = XtX_reg @ B - Xty

            # Gradient step
            B_tilde = B - step_size * gradient

            # Proximal step: SVD soft-thresholding
            if self.rank_k is not None and self.rank_k < 0.1 * min(P, V):
                # Use randomized SVD for efficiency when K is small
                try:
                    U, s, Vt = randomized_svd(
                        B_tilde,
                        n_components=self.rank_k,
                        n_oversamples=min(10, min(P, V) - self.rank_k),
                        random_state=None,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    # Fallback to full SVD if randomized fails
                    U, s, Vt = scipy.linalg.svd(
                        B_tilde, full_matrices=False, lapack_driver="gesdd"
                    )
                    U = U[:, : self.rank_k]
                    s = s[: self.rank_k]
                    Vt = Vt[: self.rank_k, :]
            else:
                # Use full SVD (reduced form)
                U, s, Vt = scipy.linalg.svd(
                    B_tilde, full_matrices=False, lapack_driver="gesdd"
                )

            # Soft-threshold singular values
            threshold = step_size * self.lambda_nuclear
            s_new = np.maximum(s - threshold, 0)

            # Apply hard rank constraint if specified
            if self.rank_k is not None and len(s_new) > self.rank_k:
                s_new = s_new[: self.rank_k]
                U = U[:, : self.rank_k]
                Vt = Vt[: self.rank_k, :]

            # Reconstruct coefficient matrix
            B_new = U @ np.diag(s_new) @ Vt

            # Compute the objective only when we need it for diagnostics.
            if self.verbose > 1:
                residual_new = Y - X @ B_new
                data_fit = 0.5 * np.linalg.norm(residual_new, "fro") ** 2
                l2_penalty = 0.5 * self.lambda_l2 * np.linalg.norm(B_new, "fro") ** 2
                nuclear_penalty = self.lambda_nuclear * np.sum(s_new)
                objective = data_fit + l2_penalty + nuclear_penalty
                self.loss_history_.append(objective)
            else:
                objective = np.nan
                self.loss_history_.append(objective)

            # Check convergence
            denominator = np.linalg.norm(B, "fro")
            if denominator < 1e-10:
                rel_change = np.linalg.norm(B_new - B, "fro")
            else:
                rel_change = np.linalg.norm(B_new - B, "fro") / denominator

            # Verbose output
            if self.verbose > 0:
                self.singular_value_history_.append(s_new.copy())

                if iteration % 10 == 0 or iteration < 5:
                    s_display = s_new[: min(20, len(s_new))]
                    s_str = ", ".join([f"{sv:.4f}" for sv in s_display])
                    if len(s_new) > 20:
                        s_str += ", ..."

                    # if self.verbose > 1, print objective and singular values; else just print iteration and rank
                    if self.verbose > 1:
                        print(
                            f"Iter {iteration:4d}: obj={objective:.6e}, "
                            f"rel_change={rel_change:.6e}, rank={np.sum(s_new > 1e-10)}"
                        )
                        print(f"           singular values: [{s_str}]")
                    else:
                        print(
                            f"Iter {iteration:4d}: rel_change={rel_change:.6e}, "
                            f"rank={np.sum(s_new > 1e-10)}"
                        )

            # Check convergence
            if rel_change < self.tol:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

            B = B_new

        # Store final results
        self.coef_ = B_new
        self.U_ = U
        self.s_ = s_new
        self.Vt_ = Vt
        self.n_iter_ = iteration + 1

        if self.verbose:
            print("\nFinal statistics:")
            print(f"  Iterations: {self.n_iter_}")
            print(f"  Final objective: {objective:.6e}")
            print(f"  Rank: {self.get_rank()}")
            print(f"  Effective rank: {self.get_effective_rank():.2f}")
            print(f"  Top 5 singular values: {s_new[: min(5, len(s_new))]}")

        return self

    def predict(self, X: np.ndarray, reduced: bool = False) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (T_new, P)
            New design matrix.

        reduced : bool, default=False
            If False, returns full-space predictions using coef_.
            If True, returns predictions using explicit low-rank decomposition
            (U_ @ diag(s_) @ Vt_), which may be more memory-efficient.

        Returns
        -------
        Y_pred : ndarray of shape (T_new, V)
            Predicted values.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{self.coef_.shape[0]} features"
            )

        if reduced:
            # Explicit low-rank prediction
            return X @ (self.U_ @ np.diag(self.s_) @ self.Vt_)
        else:
            # Full-space prediction
            return X @ self.coef_

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).

        Parameters
        ----------
        X : ndarray of shape (T, P)
            Design matrix.

        Y : ndarray of shape (T, V)
            True response matrix.

        Returns
        -------
        score : float
            R² score averaged across all outputs.
        """
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred, multioutput="uniform_average")

    def get_rank(self, threshold: float = 1e-10) -> int:
        """
        Get the rank of the coefficient matrix.

        Parameters
        ----------
        threshold : float, default=1e-10
            Singular values below this threshold are considered zero.

        Returns
        -------
        rank : int
            Number of singular values above threshold.
        """
        if not hasattr(self, "s_"):
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        return int(np.sum(self.s_ > threshold))

    def get_effective_rank(self) -> float:
        """
        Get the effective rank (participation ratio of singular values).

        The effective rank is defined as:
            sum(s)² / sum(s²)

        where s are the singular values. This provides a continuous measure
        of rank that accounts for the distribution of singular value magnitudes.

        Returns
        -------
        effective_rank : float
            Effective rank (between 1 and actual rank).
        """
        if not hasattr(self, "s_"):
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if len(self.s_) == 0 or np.sum(self.s_) < 1e-10:
            return 0.0

        return float(np.sum(self.s_) ** 2 / np.sum(self.s_**2))


def tune_hyperparameters(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_nuclear_grid: np.ndarray,
    lambda_l2_grid: np.ndarray,
    rank_k_grid: np.ndarray | list | None = None,
    cv_folds: int = 5,
    scoring: Literal["r2", "mse"] = "r2",
    verbose: bool = True,
    random_state: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Tune hyperparameters using cross-validation.

    Performs grid search over lambda_nuclear, lambda_l2, and optionally rank_k
    to find the combination that maximizes cross-validated performance.

    Parameters
    ----------
    X : ndarray of shape (T, P)
        Design matrix (predictors).

    Y : ndarray of shape (T, V)
        Response matrix (targets).

    lambda_nuclear_grid : array-like
        Grid of nuclear norm regularization values to try.

    lambda_l2_grid : array-like
        Grid of L2 regularization values to try.

    rank_k_grid : array-like or None, default=None
        Grid of rank values to try. If None, only tests rank_k=None
        (pure nuclear norm without hard rank constraint).

    cv_folds : int, default=5
        Number of cross-validation folds.

    scoring : {'r2', 'mse'}, default='r2'
        Scoring metric. 'r2' for R² score (higher is better),
        'mse' for mean squared error (lower is better).

    verbose : bool, default=True
        If True, prints progress during grid search.

    random_state : int or None, default=None
        Random state for cross-validation splitting.

    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters:
        - 'lambda_nuclear': best nuclear norm regularization
        - 'lambda_l2': best L2 regularization
        - 'rank_k': best rank (or None)
        - 'cv_score': cross-validation score with best parameters

    results_df : pd.DataFrame
        DataFrame containing all grid search results with columns:
        - lambda_nuclear, lambda_l2, rank_k
        - cv_mean: mean score across folds
        - cv_std: standard deviation across folds
        - fold_0, fold_1, ...: individual fold scores

    Examples
    --------
    >>> from task_latent.model.nn_regression import tune_hyperparameters
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(200, 50)
    >>> Y = np.random.randn(200, 100)
    >>>
    >>> best_params, results = tune_hyperparameters(
    ...     X, Y,
    ...     lambda_nuclear_grid=[0.1, 1.0, 10.0],
    ...     lambda_l2_grid=[0.01, 0.1, 1.0],
    ...     rank_k_grid=[5, 10, 20],
    ...     cv_folds=3
    ... )
    >>> print(best_params)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Setup cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Setup grid - ensure rank_k_grid is a list
    if rank_k_grid is None:
        rank_k_list = [None]
    elif isinstance(rank_k_grid, (list, tuple)):
        rank_k_list = list(rank_k_grid)
    else:
        rank_k_list = list(rank_k_grid)

    results = []

    total_combinations = (
        len(lambda_nuclear_grid) * len(lambda_l2_grid) * len(rank_k_list)
    )

    if verbose:
        print(
            f"Starting grid search over {total_combinations} parameter combinations..."
        )
        print(f"  lambda_nuclear: {list(lambda_nuclear_grid)}")
        print(f"  lambda_l2: {list(lambda_l2_grid)}")
        print(f"  rank_k: {rank_k_list}")
        print(f"  CV folds: {cv_folds}")
        print()

    combination_idx = 0

    for lambda_nuclear in lambda_nuclear_grid:
        for lambda_l2 in lambda_l2_grid:
            for rank_k in rank_k_list:
                combination_idx += 1

                if verbose:
                    print(
                        f"[{combination_idx}/{total_combinations}] "
                        f"λ_nuclear={lambda_nuclear:.3f}, "
                        f"λ_l2={lambda_l2:.3f}, "
                        f"K={rank_k if rank_k else 'None'}...",
                        end=" ",
                    )

                fold_scores = []

                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    Y_train, Y_val = Y[train_idx], Y[val_idx]

                    # Fit model
                    model = NuclearNormRegressor(
                        lambda_nuclear=lambda_nuclear,
                        lambda_l2=lambda_l2,
                        rank_k=rank_k,
                        max_iter=1000,
                        tol=1e-4,
                        verbose=False,
                    )

                    try:
                        model.fit(X_train, Y_train)

                        # Score
                        if scoring == "r2":
                            score = model.score(X_val, Y_val)
                        elif scoring == "mse":
                            Y_pred = model.predict(X_val)
                            score = -np.mean((Y_val - Y_pred) ** 2)  # negative MSE
                        else:
                            raise ValueError(f"Unknown scoring: {scoring}")

                        fold_scores.append(score)

                    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                        if verbose:
                            print(f"FAILED (fold {fold_idx}): {e}")
                        fold_scores.append(np.nan)
                        continue

                # Compute mean and std across folds
                fold_scores = np.array(fold_scores)
                valid_scores = fold_scores[~np.isnan(fold_scores)]

                if len(valid_scores) > 0:
                    cv_mean = np.mean(valid_scores)
                    cv_std = np.std(valid_scores)
                else:
                    cv_mean = np.nan
                    cv_std = np.nan

                if verbose:
                    print(f"CV: {cv_mean:.4f} ± {cv_std:.4f}")

                # Store results
                result = {
                    "lambda_nuclear": lambda_nuclear,
                    "lambda_l2": lambda_l2,
                    "rank_k": rank_k,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }

                for fold_idx, score in enumerate(fold_scores):
                    result[f"fold_{fold_idx}"] = score

                results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best parameters
    valid_results = results_df[~results_df["cv_mean"].isna()]

    if len(valid_results) == 0:
        raise RuntimeError("All parameter combinations failed during cross-validation")

    if scoring == "r2":
        best_idx = valid_results["cv_mean"].idxmax()
    else:  # mse (negative, so max is best)
        best_idx = valid_results["cv_mean"].idxmax()

    best_row = results_df.loc[best_idx]

    # Extract scalar values from Series
    rank_k_val = best_row["rank_k"]
    # Check if rank_k is not NaN
    try:
        rank_k_final = (
            int(rank_k_val.item()) if not np.isnan(rank_k_val.item()) else None
        )
    except (ValueError, TypeError):
        rank_k_final = None

    best_params = {
        "lambda_nuclear": float(best_row["lambda_nuclear"].item()),
        "lambda_l2": float(best_row["lambda_l2"].item()),
        "rank_k": rank_k_final,
        "cv_score": float(best_row["cv_mean"].item()),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print("=" * 60)

    return best_params, results_df
