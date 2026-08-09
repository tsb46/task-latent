"""
Class for iterating over fMRI data files in BIDS format.
"""

import re
from pathlib import Path
from typing import Literal

from bids import BIDSLayout, BIDSLayoutIndexer

from task_latent.constants import IBC_DATA_DIR


class FileMapper:
    """
    Maps file paths for a specific subject's fMRI data in BIDS dataset.
    """

    def __init__(self, dataset: Literal["ibc"], subject: str):
        """
        Initialize the FileMapper for a specific subject.

        Parameters
        ----------
        dataset : {'ibc'}
            The dataset name.
        subject : str
            The subject identifier.
        """
        self.dataset = dataset
        self.subject = subject
        self._initialize_layout()

    def _initialize_layout(self) -> None:
        """Initialize or refresh the BIDS layout and cached subject metadata."""
        # initialize BIDS layout
        print("Initializing BIDS layout for subject:", self.subject)
        """
        Note, for IBC dataset: the filemapper class assumes that fmri and event files 
        are in a single BIDS directory structure.
        """
        # create ignore pattern for other subjects to speed up layout initialization
        indexer = BIDSLayoutIndexer(ignore=[re.compile(f"(sub-(?!{self.subject}).*/)")])
        # The BIDSLayout initialization can be slow, especially for large datasets
        if self.dataset == "ibc":
            self.layout = BIDSLayout(
                IBC_DATA_DIR,
                indexer=indexer,
                is_derivative=True,
            )
        else:
            raise ValueError("Dataset must be 'ibc' for FileMapper.")

        # get available subjects in the dataset
        self.available_subjects = self.layout.get_subjects()
        # check whether any subjects are found
        if not self.available_subjects:
            raise RuntimeError(
                f"No subjects found in BIDS directory: {IBC_DATA_DIR if self.dataset == 'ibc' else None}"
            )

        # check if subject is valid
        if self.subject not in self.available_subjects:
            raise ValueError(f"Subject '{self.subject}' not found in dataset.")
        # get the sessions for the subject
        self.sessions = self.layout.get_sessions(subject=self.subject)
        # get the tasks for the subject
        self.tasks = self.layout.get_tasks(subject=self.subject)
        # create a nested dictionary to store iterations over tasks, sessions, and runs
        self._create_task_session_run_iter()

    def refresh_layout(self) -> None:
        """Rebuild the BIDS layout so newly created files can be discovered."""
        self._initialize_layout()

    def get_fmri_files(
        self,
        task: str,
        sessions: list[str] | None = None,
        preproc_type: Literal["orig", "final"] = "orig",
    ) -> list[str]:
        """
        Get the fMRI files from all sessions for a specific task. All fMRI files for the specified task
        and sessions will be returned, including all runs and phase encoding directions.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.
        preproc_type : {'orig', 'final'}
            The type of fMRI files to retrieve. 'orig' returns files
            with the 'preproc' description (output of fMRIPrep preprocessing).
            'final' returns files with the 'preprocfinal' description
            (output of additional final preprocessing steps).

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        # set description and extension based on preprocessing type
        extension = ".nii.gz"
        # set desc based on preproc_type
        if preproc_type == "orig":
            desc = "preproc"
        elif preproc_type == "final":
            desc = "preprocfinal"
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        fmri_files = []
        for session in sessions if sessions is not None else self.sessions:
            files = self.get_session_fmri_files(
                session, task, desc=desc, extension=extension
            )
            fmri_files.extend(files)
        return fmri_files

    def get_event_files(
        self, task: str, sessions: list[str] | None = None
    ) -> list[list[tuple[str, str]]]:
        """
        Get the event files from all sessions for a specific task. All event files for the specified task
        and sessions will be returned, including all runs and phase encoding directions.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.

        Returns
        -------
        list of list of tuple of str
            A nested list of onset and duration file path tuples (onset, duration) by session.
        """
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        event_files = []
        for session in sessions if sessions is not None else self.sessions:
            files = self.get_session_event_files(session, task)
            event_files.append(files)
        return event_files

    def get_confound_files(
        self, task: str, sessions: list[str] | None = None
    ) -> list[str]:
        """
        Get the confound time series fMRIPrep output from all sessions for a specific task. For
        each session, all confound files for the specified task will be returned, including all
        runs and phase encoding directions.

        Parameters
        ----------
        task : str
            The task identifier.
        sessions : list of str, optional
            The sessions to include. If None, all sessions are included.

        Returns
        -------
        list of str
            A list of confound file paths.
        """
        # if session is selected, ensure that it's valid
        if sessions is not None:
            for session in sessions:
                if session not in self.sessions:
                    raise ValueError(
                        f"Session '{session}' is not valid for subject '{self.subject}'."
                    )

        confound_files = []
        for session in sessions if sessions is not None else self.sessions:
            files = self.get_session_confound_files(session, task)
            confound_files.extend(files)
        return confound_files

    @staticmethod
    def get_out_directory(fp: str) -> str:
        """
        Get the output directory for a specific file path.

        Parameters
        ----------
        fp : str
            The file path.

        Returns
        -------
        str
            The output directory path.
        """
        return str(Path(fp).parent)

    def get_sessions_task(self, task: str) -> list[str]:
        """
        Get the sessions available for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        list of str
            A list of session identifiers.
        """
        sessions = self.layout.get_sessions(subject=self.subject, task=task)
        return sessions

    def get_session_event_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        ped: Literal["ap", "pa"] | None = None,
    ) -> list[str]:
        """
        Get the event files for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : Literal['ap', 'pa'] | None, optional
            The phase encoding direction. If provided, only files for this phase encoding direction will be returned

        Returns
        -------
        list of str
            List of event file paths.
        """
        bids_files = self.layout.get(
            subject=self.subject,
            session=session,
            task=task,
            suffix="events",
            extension=".tsv",
            run=run,
        )
        filenames = [f.path for f in bids_files]

        # for some reason, filtering by PhaseEncodingDirection entity does not work for some files
        # so we will filter manually after retrieving the files
        if ped is not None:
            filenames = [f for f in filenames if f"dir-{ped}" in f]

        return filenames

    def get_session_fmri_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        ped: Literal["ap", "pa"] | None = None,
        desc: Literal["preproc", "preprocfinal"] | None = "preproc",
        extension: str = ".nii.gz",
    ) -> list[str]:
        """
        Get the fMRI files for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : Literal['ap', 'pa'] | None, optional
            The phase encoding direction. If provided, only files for this phase encoding direction will be returned.
        desc : Literal['preproc', 'preprocfinal'] | None, optional
            The description entity to filter files. Defaults to 'preproc' for
            the output of fMRIPrep preprocessing. Use 'preprocfinal' for
            files that have undergone additional (final) preprocessing steps. Note,
            surface files do not have a description entity in fMRIPrep output,
        extension : str
            The file extension to filter files. Defaults to '.nii.gz' for
            volumetric fMRI files.

        Returns
        -------
        list of str
            A list of fMRI file paths.
        """
        bids_files = self.layout.get(
            subject=self.subject,
            session=session,
            task=task,
            suffix="bold",
            extension=extension,
            run=run,
            desc=desc,
        )

        filenames = [f.path for f in bids_files]

        # for some reason, filtering by PhaseEncodingDirection entity does not work for some files
        # so we will filter manually after retrieving the files
        if ped is not None:
            filenames = [f for f in filenames if f"dir-{ped}" in f]

        return filenames

    def get_session_confound_files(
        self,
        session: str,
        task: str,
        run: str | None = None,
        ped: Literal["ap", "pa"] | None = None,
    ) -> list[str]:
        """
        Get the confound time series fMRIPrep output file paths for a specific session and task.

        Parameters
        ----------
        session : str
            The session identifier.
        task : str
            The task identifier.
        run : str, optional
            The run identifier. If provided, only files for this run will be returned.
        ped : Literal['ap', 'pa'] | None, optional
            The phase encoding direction. If provided, only files for this phase encoding direction will be returned

        Returns
        -------
        list of str
            A list of confound file paths.
        """
        bids_files = self.layout.get(
            subject=self.subject,
            session=session,
            task=task,
            suffix="timeseries",
            desc="confounds",
            extension=".tsv",
            run=run,
        )
        filenames = [f.path for f in bids_files]

        # for some reason, filtering by PhaseEncodingDirection entity does not work for some files
        # so we will filter manually after retrieving the files
        if ped is not None:
            filenames = [f for f in filenames if f"dir-{ped}" in f]

        return filenames

    def get_tr(self, task: str) -> float:
        """
        Get the repetition time (TR) for a specific task.

        Parameters
        ----------
        task : str
            The task identifier.

        Returns
        -------
        float
            The repetition time (TR) in seconds.
        """
        tr = self.layout.get_tr(derivatives=True, subject=self.subject, task=task)
        return tr

    def _create_task_session_run_iter(self) -> None:
        """
        Create a nested dictionary to store iterations over tasks, sessions, and runs.
        The structure is: {task: {session: [(ped, run), ...]}}
        """
        self.tasks_iter = {}
        for task in self.tasks:
            self.tasks_iter[task] = {}
            # get sessions for task
            task_sessions = self.layout.get_sessions(subject=self.subject, task=task)
            for session in task_sessions:
                found_files = self.layout.get(
                    subject=self.subject, session=session, task=task
                )
                found_file_ents = [f.get_entities() for f in found_files]
                # get unique (PhaseEncodingDirection, run) pairs for this session and task
                unique_dir_runs = {
                    (f.get("direction"), f.get("run")) for f in found_file_ents
                }
                self.tasks_iter[task][session] = list(unique_dir_runs)
