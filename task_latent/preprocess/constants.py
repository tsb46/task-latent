"""
Constants for the preprocessing pipeline.
"""

# High-pass filter cutoff frequency for fmri
HIGHPASS = 0.01

# Full width at half maximum for Gaussian smoothing
FWHM = 4

# 3mm brain mask for IBC dataset in MNI152 space
MASK = "templates/MNI152_3mm_brain_mask_ibc.nii.gz"
