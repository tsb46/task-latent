"""
Constants for the task_latent repo.
"""

from typing import Literal

# data directory for IBC dataset
IBC_DATA_DIR: str = "data/ibc"

# 3mm MNI152 Brain mask
MASK: str = "templates/MNI152_3mm_brain_mask_ibc.nii.gz"

# Hardcoded Phase-Encoding direction labels
PHASE_ENCODING_DIRECTIONS: list[Literal["ap", "pa"]] = ["ap", "pa"]

# Tasks to include from the IBC dataset
IBC_TASKS: list[str] = [
    "ArchiStandard",
    "ArchiSpatial",
    "ArchiSocial",
    "ArchiEmotional",
    "HcpEmotion",
    "HcpGambling",
    "HcpMotor",
    "HcpLanguage",
    "HcpRelational",
    "HcpSocial",
    "HcpWm",
    "RSVPLanguage",
    "MTTWE",
    "MTTNS",
    "PreferenceFood",
    "PreferencePaintings",
    "PreferenceFaces",
    "PreferenceHouses",
    "TheoryOfMind",
    "EmotionalPain",
    "PainMovie",
    "VSTM",
    "Enumeration",
    "Self",
    "Bang",
    "WedgeClock",
    "WedgeAnti",
    "ContRing",
    "ExpRing",
    "Lec2",
    "Audi",
    "Visu",
    "Lec1",
    "MVEB",
    "MVIS",
    "Moto",
    "MCSE",
    "Audio",
    "Attention",
    "StopSignal",
    "TwoByTwo",
    "SelectiveStopSignal",
    "Stroop",
    "ColumbiaCards",
    "DotPatterns",
    "WardAndAllport",
    "BiologicalMotion1",
    "BiologicalMotion2",
    "MathLanguage",
    "SpatialNavigation",
    "EmoMem",
    "EmoReco",
    "StopNogo",
    "Catell",
    "FingerTapping",
    "VSTMC",
    "RewProc",
    "NARPS",
    "FaceBody",
    "Scene",
    "BreathHolding",
    "Checkerboard",
    "FingerTap",
    "ItemRecognition",
    "VisualSearch",
    "Color",
    "Motion",
]
