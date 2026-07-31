"""
Constants for the preprocessing pipeline.
"""

# High-pass filter cutoff frequency for fmri
HIGHPASS = 0.01

# Full width at half maximum for Gaussian smoothing
FWHM = 4

# 3mm brain mask for IBC dataset
# copied from:
# https://github.com/individual-brain-charting/public_analysis_code/tree/master/ibc_data
MASK = "templates/gm_mask_3mm.nii.gz"
