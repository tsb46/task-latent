"""
Class for managing and loading preprocessed dataset files for
a given subject in the IBC dataset.
"""

import json
import math
from dataclasses import dataclass
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask, unmask
from scipy.stats import zscore

from task_latent.constants import MASK
from task_latent.io.file import FileMapper


@dataclass(frozen=True)
class ScanMetadata:
    """
    Metadata for each fMRI task scan.

    Attributes
    ----------
    label: str
        Task label
    conditions: str
        Task conditions after grouping and exclusion - as specified in the `ibc_conditions.json` file
    tr: float
        Repetition time of the fMRI scan in seconds
    n_frames: int
        Number of frames in the fMRI scan
    ped: str
        Phase encoding
    session: str
        Session label for task
    run: str, optional
        Run label, if multiple runs for task

    """

    label: str
    conditions: list[str]
    tr: float
    n_frames: int
    ped: Literal["pa", "ap"]
    session: str
    run: str | None


# Dataset class
@dataclass(frozen=True)
class Dataset:
    """
    Loaded data for a given subject in the IBC dataset.

    Attributes
    -----------
    fmri: list[np.ndarray]
        List of 2D (time, voxel) matrices of fMRI scans
    events: list[pd.DataFrame]
        List of BIDs-formatted event Pandas dataframes. Order of dataframes matches
        the scans in the fMRI list.
    scan_metadata: list[ScanMetadata]
        List of scan metadata. Order of metadata objects matches the scans in the
        fMRI list.
    task_index: dict[str, list[int]]
        Dict containing the indices of each run for the same task in the
        data lists (fmri, events, scan_metadata)
    subject: str
        Subject label
    normalize: bool
        Whether the fMRI scans have been z-score normalized (along the time dimension)

    """

    # data
    fmri: list[np.ndarray]
    events: list[pd.DataFrame]
    scan_metadata: list[ScanMetadata]
    # task to list index
    task_index: dict[str, list[int]]
    # load parameters
    subject: str
    normalize: bool


class DataLoader:
    """
    Class for managing and loading preprocessed dataset files for a given subject in
    the IBC dataset. FMRI dataset is returned as a 2D array  list
    of 2D arrays (one per session) depending on the `concatenate` parameter. Event data is
    returned as a pandas DataFrame or list of DataFrames (one per session) depending on the
    `concatenate` parameter.

    Parameters
    ----------
    dataset : str
        The dataset name. Currently only supports "ibc".
    subject : str
        The subject identifier.

    Attributes
    ----------
    file_mapper : FileMapper
        An instance of the FileMapper class for mapping file paths associated with the subject.
    tasks : list[str]
        A list of available tasks for the subject.
    sessions : list[str]
        A list of available sessions for the subject.
    condition_metadata : dict
        A dictionary containing condition metadata for tasks, loaded from `ibc_conditions.json` file.
    mask : nib.Nifti1Image
        A brain mask in the same space as the fMRI data, loaded from a NIfTI file. Defined in `task_latent.constants.MASK`.

    Methods
    -------
    load(task: list[str] | None = None, normalize: bool = True, verbose: bool = True) -> Dataset
        Load the preprocessed dataset files for the subject, returning a Dataset object.
    events_to_df(fp_event: str, ped: str, session: str, drop_time: float | None = None, condition_grouper: dict | None = None, condition_ignore: list[str] | None = None) -> pd.DataFrame
        Load bids-formatted event file to a pandas DataFrame
    load_fmri(fp: str, normalize: bool = False, convert_to_2d: bool = True, verbose: bool = True) -> np.ndarray
        Load the preprocessed fMRI data from a NIfTI file.
    to_4d(fmri_data: np.ndarray) -> nib.Nifti1Image
        Convert a 2D fMRI array (time x voxels) back to a 4D NIfTI image using the provided mask.

    """

    def __init__(self, dataset: Literal["ibc"], subject: str):
        """
        Initialize the DataLoader class for a specific subject in the IBC dataset.

        Parameters
        ----------
        dataset : str
            The dataset name. Currently only supports "ibc".
        subject : str
            The subject identifier.
        """
        # ensure dataset is 'ibc'
        if dataset != "ibc":
            raise ValueError(f"Dataset '{dataset}' is not supported for DataLoader.")
        # ensure subject is passed
        if not subject:
            raise ValueError("Subject label must be provided for DataLoader.")
        self.subject = subject
        self.dataset = dataset
        # map file paths associated to subject
        self.file_mapper = FileMapper(dataset=dataset, subject=subject)
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks
        # get available sessions from mapper
        self.sessions = self.file_mapper.sessions
        # load condition metadata for tasks from JSON file
        # Conditions for IBC tasks
        with open("task_latent/dataset/ibc_conditions.json", "r") as f:
            try:
                # load JSON and convert nulls to NaN for consistency
                self.condition_metadata = json.load(f, object_hook=convert_none_to_nan)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error loading IBC conditions JSON file: {e}") from e

        # attach mask to instance
        self.mask = nib.load(MASK)

    def load(
        self,
        tasks: list[str] | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> Dataset:
        """
        Load the preprocessed dataset files for the subject.

        Parameters
        ----------
        tasks : list[str] | None, optional
            The task identifier(s) to load. If None, all tasks will be loaded. Default is None
        normalize : bool, optional
            Whether to normalize (z-score) the fMRI data along the time dimension. Default is True.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        # if task is not None, ensure it's an available task
        if tasks is not None:
            if isinstance(tasks, str):
                tasks = [tasks]
            for t in tasks:
                if t not in self.tasks:
                    raise ValueError(
                        f"Task '{t}' is not available for subject '{self.subject}'."
                    )
        else:
            tasks = self.tasks

        # initialize data list/dicts for tasks
        fmri = []
        events = []
        scan_metadata = []
        # counter for indexing tasks in the data lists
        task_index = {
            t: []
            for t in tasks
            if self.condition_metadata.get(t, {}).get("keep", False)
        }
        scan_indx = 0
        # iterate through tasks to load data
        for task in tasks:
            if verbose:
                print(f"Loading data for task '{task}'...")

            # check whether the task is marked to be kept in the conditions JSON file
            task_metadata = self.condition_metadata.get(task, {})
            if not task_metadata.get("keep", False):
                if verbose:
                    print(
                        f"  Skipping task '{task}' as it is marked to be ignored in the conditions JSON file."
                    )
                continue
            # get tr for task
            tr = self.file_mapper.get_tr(task)

            # get conditions for task
            try:
                condition_metadata = self.condition_metadata[task]
            except KeyError:
                raise ValueError(
                    f"No condition metadata found for task '{task}' in IBC conditions JSON file."
                )
            conditions = condition_metadata.get("condition", [])
            # get grouping of conditions, if any
            condition_grouper = condition_metadata.get("condition_grouper", None)
            if condition_grouper is None:
                raise ValueError(f"No condition_grouper found for task '{task}'.")
            # get conditions to ignore, if any
            condition_ignore = condition_metadata.get("condition_ignore", [])

            # get (run, phase-encoding direction) pairs available for task
            runs_ped_pairs = self.file_mapper.tasks_iter[task]

            # get sessions avaialble for task
            task_sessions = self.file_mapper.get_sessions_task(task)
            # check if sessions are available for task
            if len(task_sessions) == 0:
                raise ValueError(
                    f"No sessions found for task '{task}' and subject '{self.subject}'."
                )

            # load files for each session
            for session in task_sessions:
                if verbose:
                    print(f"  Loading session '{session}'...")
                # loop through runs and phase encoding directions for this session
                for ped, run in runs_ped_pairs[session]:
                    if verbose:
                        print(f"    Loading ped '{ped}' run '{run}'...")
                    fmri_files = self.file_mapper.get_session_fmri_files(
                        session, task, ped=ped, run=run, desc="preprocfinal"
                    )
                    # if no fMRI file is found, raise error
                    if len(fmri_files) == 0:
                        raise ValueError(
                            f"No fMRI file found for session '{session}' "
                            f"and task '{task}' and phase encoding direction '{ped}' "
                            f"and run {run}."
                        )
                    elif len(fmri_files) > 1:
                        # raise error if multiple fMRI files are found
                        raise ValueError(
                            f"Multiple fMRI files found for session '{session}' "
                            f"and task '{task}' and phase encoding direction '{ped}' "
                            f"and run {run}."
                        )
                    # load fMRI file into 2D array or 4D image
                    fmri_data = self.load_fmri(
                        fmri_files[0],
                        normalize=normalize,
                        verbose=verbose,
                    )
                    # append data to dataset
                    fmri.append(fmri_data)

                    # load event files
                    event_file = self.file_mapper.get_session_event_files(
                        session, task, ped=ped, run=run
                    )
                    if len(event_file) == 0:
                        raise ValueError(
                            f"Warning: No event file found for session '{session}' "
                            f"and task '{task}' and phase encoding direction '{ped}' "
                            f"and run {run}."
                        )
                    elif len(event_file) > 1:
                        raise ValueError(
                            f"Multiple event files found for session '{session}' "
                            f"and task '{task}' and phase encoding direction '{ped}' "
                            f"and run {run}."
                        )
                    # convert event file to dataframe
                    event_df = self.events_to_df(
                        fp_event=event_file[0],
                        task=task,
                        ped=ped,
                        session=session,
                        conditions=conditions,
                        condition_grouper=condition_grouper,
                        condition_ignore=condition_ignore,
                    )
                    events.append(event_df)

                    # create scan metadata object
                    metadata = ScanMetadata(
                        label=task,
                        conditions=conditions,
                        ped=ped,
                        tr=tr,
                        n_frames=fmri_data.shape[0],
                        session=session,
                        run=run,
                    )
                    scan_metadata.append(metadata)

                    # append scan index to task_index dict
                    task_index[task].append(scan_indx)

        if verbose:
            print(f"Data loading complete for subject '{self.subject}'")

        return Dataset(
            fmri=fmri,
            events=events,
            scan_metadata=scan_metadata,
            task_index=task_index,
            subject=self.subject,
            normalize=normalize,
        )

    def events_to_df(
        self,
        fp_event: str,
        task: str,
        ped: str,
        session: str,
        conditions: list[str],
        condition_grouper: dict | None = None,
        condition_ignore: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load BIDS-formatted event files as a pandas DataFrame.

        Parameters
        ----------
        fp_event : str
            The path to the event file.
        task: str
            Task label
        session : str
            The session identifier.
        ped: str
            The phase encoding direction.
        conditions: list[str]
            The conditions for the task (after grouping and exclusion).
        condition_grouper : dict, optional
            A dictionary mapping original condition names to grouped condition names.
            If provided, conditions will be grouped accordingly. Default is None.
        condition_ignore : List[str], optional
            A list of condition names to ignore. If provided, events with these condition names
            will be removed from the DataFrame. Default is None.

        Returns
        -------
        pd.DataFrame
            The event data as a pandas DataFrame.
        """
        event_df = pd.read_csv(fp_event, delimiter="\t")
        event_df = event_df.sort_values(by="onset")
        # insert session column
        event_df.insert(0, "session", session)
        # insert ped column
        event_df.insert(1, "ped", ped)

        # if condition_grouper is provided, map conditions
        if condition_grouper is not None:
            event_df["trial_type"] = event_df["trial_type"].replace(condition_grouper)
        # if condition_ignore is provided, remove those conditions
        if condition_ignore is not None:
            event_df = event_df[
                ~event_df["trial_type"].isin(condition_ignore)
            ].reset_index(drop=True)

        # ensure there are no conditions in the event dataframe missing in conditions list
        ev_conditions = event_df["trial_type"].unique()
        for cond in ev_conditions:
            if cond not in conditions:
                raise ValueError(
                    f"The event dataframe for task {task} contains a condition "
                    f"not specified in `ibc_conditions.json`: {cond}"
                )
        # ensure that all conditions in the conditions list are present in the event dataframe
        for cond in conditions:
            if cond not in ev_conditions:
                raise ValueError(
                    f"The event dataframe for task {task} is missing a condition "
                    f"specified in `ibc_conditions.json`: {cond}"
                )

        return event_df

    def load_fmri(
        self,
        fp: str,
        normalize: bool = False,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Load the preprocessed fMRI data from a NIfTI file and return as a 2D numpy
        array (time by voxels).

        Parameters
        ----------
        fp : str
            NIfTI file path.
        mask_img : nib.nifti1.Nifti1Image
            Brain mask in the same space as fp.

        Returns
        ---------
        np.ndarray

        """
        img = nib.load(fp)
        # check img type
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError(
                f"Expected Nifti1Image for fMRI data. Got {type(img)} instead."
            )
        # apply mask to get 2D data
        data_2d = apply_mask(img, self.mask)  # shape: time x voxels

        # check for any NaNs
        if np.isnan(data_2d).any():
            raise ValueError(
                f"fMRI file {fp} contains null values. Check mask or preprocessing outputs"
            )

        if normalize:
            data_2d = zscore(data_2d, axis=0)
            data_2d = np.array(data_2d)
            # check for the introduction of NaNs
            nan_mask = np.isnan(data_2d).any(axis=0)
            n_nan_voxels = nan_mask.sum()
            # print out any nulls
            if n_nan_voxels > 0 and verbose:
                print(
                    f"     Warning: z-scoring introduced {n_nan_voxels} NaN voxels. Will be filled with zero"
                )
            data_2d[np.isnan(data_2d)] = 0.0

        return data_2d

    def to_4d(self, fmri_data: np.ndarray) -> nib.Nifti1Image:
        """
        Unmask a 2D fMRI array (time x voxels) back to a 4D NIfTI using the provided mask.
        """
        fmri_4d_img = unmask(fmri_data, mask_img=self.mask)  # type: ignore
        assert isinstance(fmri_4d_img, nib.Nifti1Image), (
            "to_4d did not return a Nifti1Image."
        )
        return fmri_4d_img


def convert_none_to_nan(obj):
    """
    Recursively convert None values in a nested structure (dicts, lists) to NaN.
    """
    if isinstance(obj, dict):
        return {k: convert_none_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_none_to_nan(element) for element in obj]
    elif obj is None:
        return math.nan
    return obj
