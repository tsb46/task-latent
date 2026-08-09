"""
Constants for the task_latent repo.
"""

from typing import Literal

# data directory for IBC dataset
IBC_DATA_DIR = "data/ibc"

# 3mm MNI152 Brain mask
MASK = "templates/MNI152_3mm_brain_mask_ibc.nii.gz"

# Hardcoded Phase-Encoding direction labels
PHASE_ENCODING_DIRECTIONS: list[Literal["ap", "pa"]] = ["ap", "pa"]
