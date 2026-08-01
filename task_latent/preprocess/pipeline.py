"""
Additional preprocessing pipeline applied to minimally preprocessed IBC data.

Steps:

1) Detrending
2) High-pass filtering (> 0.01 Hz)
3) Standardize signal (psc)
4) Smoothing
"""

import os
from pathlib import Path
from typing import Literal

import nibabel as nib
from joblib import Parallel, delayed
from nilearn.image import clean_img, smooth_img
from nilearn.masking import apply_mask, unmask

from task_latent.io.file import FileMapper

# Preprocessing parameters
from task_latent.preprocess.constants import (
    FWHM,  # smoothing FWHM
    HIGHPASS,  # high-pass filter cutoff frequency
    MASK,  # 3mm brain mask
)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for IBC minimilly preprocessed data.

    This class provides methods to perform additional preprocessing steps on minimally preprocessed IBC data, including
    detrending, high-pass filtering, standardization, and smoothing.
    """

    def __init__(
        self,
        dataset: Literal["ibc"],
        subject: str,
    ) -> None:
        """Initialize the preprocessing pipeline for a specific dataset and subject.

        Parameters
        ----------
            dataset (Literal['ibc']): The dataset identifier.
            subject (str): The subject identifier.
        """
        self.subject = subject
        self.dataset = dataset

        # map file paths associated to subject
        if dataset == "ibc":
            self.file_mapper = FileMapper(dataset, subject)
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported.")
        # get available tasks from mapper
        self.tasks = self.file_mapper.tasks

    def preprocess(
        self,
        task: str | None = None,
        sessions: list[str] | None = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Perform additional preprocessing pipeline for minimally preprocessed IBC data.

        Parameters
        ----------
        task : str or None, optional
            The task identifier. If no task identifier is provided, all tasks will be processed. Defaults to None.
        sessions : list of str or None, optional
            The session identifiers. If no session identifiers are provided, all sessions will be processed. Defaults to None.
        n_jobs : int, optional
            Number of parallel jobs for fMRI file processing. ``1`` runs sequentially (default).
            ``-1`` uses all available CPU cores. Passed directly to ``joblib.Parallel``.
            Defaults to 1.
        verbose : bool, optional
            If True, print progress messages. Defaults to True.
        """

        # if task is not None, ensure it's an available task
        if task is not None:
            if task not in self.tasks:
                raise ValueError(
                    f"Task '{task}' is not available for subject '{self.subject}'."
                )
            tasks_to_process = [task]
        else:
            tasks_to_process = self.tasks
        if verbose:
            print(f"Processing tasks for subject '{self.subject}': {tasks_to_process}")

        # loop through each task and process scans and physio from all sessions (and runs)
        for task_proc in tasks_to_process:
            if verbose:
                print(
                    f"Processing task '{task_proc}' for subject '{self.subject}' "
                    f"and sessions '{sessions if sessions is not None else 'all'}'..."
                )

            # get TR from BIDS layout - scan metadata is same for all runs of a task
            tr = self.file_mapper.layout.get_tr(derivatives=True, task=task_proc)

            # get fmri files for task
            fmri_files = self.file_mapper.get_fmri_files(task_proc, sessions=sessions)

            # Pre-compute transform files and output paths for each fMRI file in
            # the main process (serial) so that PyBIDS layout access and
            # file-mapper calls stay single-threaded before work is handed off
            # to parallel workers.
            fmri_job_args = []
            for fmri_file in fmri_files:
                output_path = self._get_fmri_output_path(fmri_file)
                fmri_job_args.append((fmri_file, output_path))

            # Process fMRI files — parallel when n_jobs != 1, sequential otherwise.
            Parallel(n_jobs=n_jobs)(
                delayed(_process_single_fmri)(
                    fmri_file=args[0],
                    output_path=args[1],
                    tr=tr,
                    mask=MASK,
                    fwhm=FWHM,
                    highpass=HIGHPASS,
                    detrend=True,
                    standardize=True,
                    verbose=verbose,
                )
                for args in fmri_job_args
            )

            if verbose:
                print(
                    f"Finished processing task '{task_proc}' for subject '{self.subject}'."
                )

    def _get_fmri_output_path(
        self,
        fmri_file: str,
    ) -> str:
        """Compute the output file path for a preprocessed fMRI file.

        Called in the main process before the parallel fMRI loop so that each
        worker receives a ready-made output path and does not need access to
        ``self``.
        """
        output_dir = self.file_mapper.get_out_directory(fmri_file)
        file_orig_name = os.path.basename(fmri_file)
        output_name = _make_desc_preprocfinal_name(file_orig_name)
        return f"{output_dir}/{output_name}"


def func_volume_pipeline(
    func_fp: str,
    tr: float,
    brain_mask_fp: str,
    fwhm: float,
    highpass: float | None,
    detrend: bool = True,
    standardize: bool = True,
    verbose: bool = True,
) -> nib.Nifti1Image:
    """
    Functional volume pipeline for processing functional MRI data.

    Preprocessing steps:

    (Perform transforms to standard space if to_std is True, using func_to_std)

    1) Detrending (clean_img)
    2) High-pass filtering (> 0.01 Hz) (clean_img)
    3) Standardization (clean_img)
    4) Smoothing (smooth_img)

    Parameters
    ----------
    func_fp : str
        The file path to the functional MRI data.
    tr : float
        The repetition time (TR) of the fMRI data.
    brain_mask_fp : str
        The file path to the brain mask.
    fwhm : float
        The full width at half maximum (FWHM) for spatial smoothing.
    highpass : float | None
        The high-pass filter cutoff frequency in Hz.
    detrend : bool, optional
        Whether to apply detrending, by default True.
    standardize : bool, optional
        Whether to apply standardization (z-score), by default True.

    Returns
    -------
    nib.Nifti1Image
        The processed functional MRI data.
    """

    func_fp_p = Path(func_fp)
    if not func_fp_p.exists():
        raise FileNotFoundError(f"Functional file not found: {func_fp}")

    mask_fp_p = Path(brain_mask_fp)
    if not mask_fp_p.exists():
        raise FileNotFoundError(f"Brain mask file not found: {brain_mask_fp}")

    if tr <= 0:
        raise ValueError(f"tr must be > 0, got {tr}")
    if highpass is not None and highpass < 0:
        raise ValueError(f"highpass must be >= 0, got {highpass}")
    if fwhm is None:
        raise ValueError("fwhm must not be None")

    # Load functional MRI data
    func_img = nib.load(func_fp)

    # load mask
    mask_img = nib.load(brain_mask_fp)

    # ensure correct types and dimensionalities
    if not isinstance(func_img, nib.Nifti1Image):
        raise TypeError(f"Loaded fMRI data is not a Nifti1Image: {type(func_img)}")
    if not isinstance(mask_img, nib.Nifti1Image):
        raise TypeError(f"Loaded mask is not a Nifti1Image: {type(mask_img)}")
    if mask_img.ndim != 3:
        raise ValueError(f"brain_mask_fp must be 3D, got shape {mask_img.shape}")
    if func_img.ndim != 4:
        raise ValueError(f"func_fp must be 4D (x,y,z,t), got shape {func_img.shape}")

    # Make sure func and mask grids match in XYZ.
    if func_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            "Functional image and mask have different spatial shapes. "
            f"func shape[:3]={func_img.shape[:3]} vs mask shape[:3]={mask_img.shape[:3]}. "
            "Provide a mask in the same space/resolution as func_fp."
        )

    # using the clean_img function to detrend, high-pass filter, and standardize the signal
    func_img_proc = clean_img(
        func_img,
        detrend=detrend,
        standardize="zscore_sample",
        high_pass=highpass,
        mask_img=mask_img,
        t_r=tr,
    )

    # Apply spatial smoothing
    if float(fwhm) > 0:
        func_img_proc = smooth_img(func_img_proc, fwhm=fwhm)

    # Mask out smoothed data to ensure non-brain voxels are zero
    func_data_masked = apply_mask(func_img_proc, mask_img)
    func_img_proc = unmask(func_data_masked, mask_img)

    return func_img_proc


def _process_single_fmri(
    fmri_file: str,
    output_path: str,
    tr: float,
    mask: str,
    fwhm: float,
    highpass: float,
    detrend: bool,
    standardize: bool,
    verbose: bool = True,
) -> None:
    """Process a single fMRI file and write the result to *output_path*.

    Module-level function (no ``self``) so it can be pickled by ``joblib``
    and executed in a worker process.
    """
    if verbose:
        print(f"Preprocessing fMRI file: {fmri_file}")

    fmri_proc = func_volume_pipeline(
        func_fp=fmri_file,
        tr=tr,
        brain_mask_fp=mask,
        fwhm=fwhm,
        highpass=highpass,
        detrend=detrend,
        standardize=standardize,
    )
    if not isinstance(fmri_proc, nib.Nifti1Image):
        raise TypeError(
            f"Expected NIfTI image for volume output, got {type(fmri_proc)}"
        )
    nib.save(fmri_proc, output_path)


def _make_desc_preprocfinal_name(name: str) -> str:
    """Create a BIDS-ish output filename with `desc-preprocfinal`."""
    # replace `desc-preproc` with `desc-preprocfinal` in the base name
    source_desc = "desc-preproc"
    new_name = name.replace(source_desc, "desc-preprocfinal", 1)
    return new_name
