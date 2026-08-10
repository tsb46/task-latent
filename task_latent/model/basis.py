"""
Spline basis for modeling task-evoked hemodynamic response functions.
"""

import warnings
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy.interpolate import interp1d
from scipy.linalg import null_space


class HRFBasis:
    """
    Spline basis for modeling task-evoked hemodynamic response functions.

    The basis is defined over post-stimulus time in seconds and is applied
    by convolving task regressors with spline basis functions.

    Parameters
    ----------
    duration_sec : float
        Duration of the HRF response window.

    zero_at_onset : bool
        Constrain the spline space such that any linear combination of basis functions evaluates to zero
        at the minimum bound of the response window.

    knots_per_sec : float
        Approximate density of spline basis functions. This is used to determine the number of knots for the spline basis functions.

    knot_spacing : {"uniform", "geometric"}
        Strategy for distributing knots over the response window before any
        basis-type-specific adjustment.

    geometric_alpha : float
        Controls concentration of knots near stimulus onset.
        Larger values place more knots early in the HRF.

    basis_type : {"bs", "cr"}
        B-spline or cubic regression spline. For ``bs``, the knot sequence
        includes the response-window endpoints. For ``cr``, Formulaic expects
        only the interior knots, so the endpoints are treated as boundaries and
        omitted from the knot list passed to ``cr()``.

    Usage:
        >>> hrf_basis = HRFBasis(duration_sec=30.0, knot_spacing="geometric", basis_type="cr")
        >>> hrf_basis.create()
        >>> design_matrix = hrf_basis.project(task_regressor, tr=2.0)
    """

    def __init__(
        self,
        duration_sec: float,
        zero_at_onset: bool = True,
        knots_per_sec: float = 0.2,
        knot_spacing: Literal["uniform", "geometric"] = "uniform",
        geometric_alpha: float = 2.0,
        basis_type: Literal["bs", "cr"] = "cr",
    ):

        if basis_type not in ("bs", "cr"):
            raise ValueError("basis_type must be 'bs' or 'cr'")

        if knot_spacing not in ("uniform", "geometric"):
            raise ValueError("knot_spacing must be 'uniform' or 'geometric'")

        self.duration_sec = duration_sec
        self.knots_per_sec = knots_per_sec
        self.zero_at_onset = zero_at_onset
        self.knot_spacing = knot_spacing
        self.geometric_alpha = geometric_alpha
        self.basis_type = basis_type

    def _create_knots(self):

        # Get number of basis functions based on duration and knots per second.
        n_basis = max(3, int(np.ceil(self.duration_sec * self.knots_per_sec)))

        # Create knots for spline basis functions.
        if self.knot_spacing == "uniform":
            knots = np.linspace(0, self.duration_sec, n_basis)

        else:
            # Create knots with geometric spacing to concentrate basis functions near stimulus onset.
            x = np.linspace(0, 1, n_basis)
            knots = (
                self.duration_sec
                * (np.exp(self.geometric_alpha * x) - 1)
                / (np.exp(self.geometric_alpha) - 1)
            )

        # Formulaic's cr() expects inner knots only; bs() accepts the full knot sequence.
        if self.basis_type == "cr" and len(knots) > 2:
            knots = knots[1:-1]

        self.knots = knots

        return knots

    def create(self, dt: float = 0.1, extrapolation: str = "extend"):
        """
        Create the spline basis functions over the specified duration.

        Parameters
        ----------
        dt : float, default=0.1
            Time step for sampling the basis functions in seconds.
        extrapolation : str, default="extend"
            Extrapolation method for the spline basis functions, options
            specified in the formulaic documentation. Default is "extend" to
            allow evaluation of the basis functions at the minimum and maximum bounds of the response window.
        """

        self.dt = dt
        self.extrapolation = extrapolation

        self.times = np.arange(0, self.duration_sec + dt, dt)

        knots = self._create_knots()

        self.basis = np.asarray(
            model_matrix(
                f"{self.basis_type}(x, knots=({','.join(map(str, knots))}), extrapolation='{self.extrapolation}') - 1",
                pd.DataFrame({"x": self.times}),
            )
        )

        # create a costrained basis such that any linear combination of basis functions
        #  evaluates to zero at the minimum bound of the response window
        if self.zero_at_onset:
            self.basis = self._apply_zero_constraint(self.basis)

        self._n_basis = self.basis.shape[1]

        return self

    def project(self, X, tr: float, fill_value: float = 0) -> np.ndarray:
        """
        Project a stimulus time course onto the spline basis functions.

        Parameters
        ----------
        X : np.ndarray
            One-dimensional stimulus time course with shape (n_timepoints,).
        tr : float
            Repetition time of the fMRI acquisition in seconds.
        fill_value : float, default=0
            Value used to fill samples before the start of the time series.
        """

        if not hasattr(self, "basis"):
            raise RuntimeError("Call create() before project().")

        # HRF lag grid (seconds)
        hrf_times = self.times

        # interpolate basis onto TR grid
        interp = interp1d(
            hrf_times,
            self.basis,
            axis=0,
            bounds_error=False,
            fill_value=0,
        )

        basis_tr = interp(np.arange(0, self.duration_sec + tr, tr))

        # build lag matrix using TR-based lags
        lags = np.arange(basis_tr.shape[0])

        lag_matrix = lag_mat(X, lags, fill_val=fill_value)

        return lag_matrix @ basis_tr

    def plot(self, ax=None, labels: list[str] | None = None):
        """
        Plot the spline basis functions as line plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If omitted, a new figure and axis are created.
        labels : list[str], optional
            Custom labels for the basis functions.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis containing the plot.
        """

        if not hasattr(self, "basis"):
            raise RuntimeError("Call create() before plot().")

        if ax is None:
            _, ax = plt.subplots()

        n_basis = self.basis.shape[1]
        if labels is None:
            labels = [f"basis {i + 1}" for i in range(n_basis)]
        elif len(labels) != n_basis:
            raise ValueError("labels must match the number of basis functions.")

        cmap = mpl.colormaps["tab20"].resampled(n_basis)  # type: ignore

        for i in range(n_basis):
            ax.plot(self.times, self.basis[:, i], color=cmap(i), label=labels[i])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Basis value")
        ax.set_title("HRF spline basis")
        ax.legend(frameon=False)

        return ax

    def _apply_zero_constraint(self, basis: np.ndarray) -> np.ndarray:
        """
        Apply a zero-at-onset constraint to the spline basis functions via
        null space projection.
        """
        constraint = basis[0:1, :]

        N = null_space(constraint)

        return basis @ N


def lag_mat(
    X: np.ndarray,
    lags: np.ndarray,
    fill_val: float = 0.0,
) -> np.ndarray:
    """
    Create a positive-lagged design matrix from a stimulus time course.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional stimulus time course with shape (n_timepoints,).

    lags : np.ndarray
        Positive integer lags (in TRs). Lag 0 corresponds to the original
        time course.

    fill_val : float, default=0.0
        Value used to fill samples before the start of the time series.

    Returns
    -------
    lagged : np.ndarray
        Lag matrix with shape (n_timepoints, n_lags).

        Column j contains X shifted by lags[j] TRs:

            lagged[t, j] = X[t - lags[j]]

    Notes
    -----
    This function is intended for distributed lag / HRF basis modeling,
    where each column represents the stimulus history at a different
    post-stimulus delay.
    """

    X = np.asarray(X)

    if X.ndim != 1:
        raise ValueError("X must be a one-dimensional array.")

    lags = np.asarray(lags)

    if np.any(lags < 0):
        raise ValueError("Only positive lags are allowed.")

    if not np.all(lags.astype(int) == lags):
        raise ValueError("lags must contain integer TR offsets.")

    lags = lags.astype(int)

    n_time = X.shape[0]

    lagged = np.full(
        (n_time, len(lags)),
        fill_val,
        dtype=float,
    )

    for i, lag in enumerate(lags):
        if lag == 0:
            lagged[:, i] = X

        else:
            lagged[lag:, i] = X[:-lag]

    return lagged


def convolve_event_with_basis(
    event_df: pd.DataFrame,
    tr: float,
    n_frames: int,
    basis: HRFBasis,
    per_condition_normalization: bool = True,
    per_task_normalization: bool = False,
    fill_value=0.0,
    slicetime_ref: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Convolve a stimulus time course with an HRF basis. Takes a BIDS-formatted
    event dataframe with columns ["onset", "duration", "amplitude"] and returns a
    time-by-basis design matrix.

    Parameters
    ----------
    event_df : pd.DataFrame
        DataFrame containing event information with columns ["onset", "duration", "amplitude"].

    tr : float
        fMRI repetition time in seconds.

    n_frames : int
        Number of time points in the fMRI time series.

    basis : HRFBasis
        HRF basis object.

    per_condition_normalization : bool, default=True
        Whether to normalize (frobenius norm) the convolved regressors for each condition separately. This should be
        applied to ensure that the model is not biased toward longer conditions with more events. Default is True.

    per_task_normalization : bool, default=False
        Whether to normalize (frobenius norm) the convolved regressors within each the task. This can be applied to ensure
        that the model is not biased toward tasks with more conditions. Default is False.

    fill_value : float, default=0.0
        Value used to fill samples before the start of the time series.

    slicetime_ref : float, default=0.0
        Reference time for slice timing correction. This is used to adjust the onset times of events
        to account for the fact that different slices in an fMRI volume are acquired at different times
        within a TR. The value should be between 0 and 1, where 0 corresponds to the first slice and 1 corresponds to the last slice.
        The default value of 0 corresponds to the first slice.

    Returns
    -------
    convolved : tuple[np.ndarray, list[str]]
        Time-by-basis design matrix and list of condition names ({label}_knot_{i}).
    """
    # enure basis is created
    if not hasattr(basis, "basis"):
        raise RuntimeError("Call basis.create() before convolve_event_with_basis().")
    # get time samples of functional scan based on slicetime reference
    frametimes = np.linspace(
        slicetime_ref, (n_frames - 1 + slicetime_ref) * tr, n_frames
    )
    # create boxcar stimulus time course from event information
    # ignore warning about unexpected columns in event_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boxcar_matrix = make_first_level_design_matrix(
            frametimes,
            events=event_df,
            hrf_model=None,
            drift_model=None,
            high_pass=None,
        )
    # nilearn inserts a column for the constant term, which we don't need
    if "constant" in boxcar_matrix.columns:
        boxcar_matrix = boxcar_matrix.drop(columns=["constant"])

    # loop through each trial type and convolve with basis
    convolved_list = []
    labels_list = []
    trial_names = boxcar_matrix.columns.tolist()
    for trial_type in trial_names:
        boxcar = boxcar_matrix[trial_type].values
        # project boxcar onto basis functions
        convolved = basis.project(boxcar, tr=tr, fill_value=fill_value)
        # if per_condition_normalization, normalize convolved regressors for each condition separately
        if per_condition_normalization:
            convolved /= np.linalg.norm(convolved, ord="fro")

        convolved_list.append(convolved)
        labels_list.extend(
            [f"{trial_type}_knot_{i}" for i in range(convolved.shape[1])]
        )

    # horizontally stack convolved regressors
    convolved_stacked = np.hstack(convolved_list)
    # if per_task_normalization, normalize convolved regressors within each task
    if per_task_normalization:
        convolved_stacked /= np.linalg.norm(convolved_stacked, ord="fro")

    return convolved_stacked, labels_list
