"""CBC curriculum domain constants and track-to-pathway mappings.

This module is the single source of truth for every curriculum entity
used across the prototype: subjects, grade-term sequences, tracks,
pathways, and the weighted subject profiles that define each track.

The track weight dictionaries are the domain-expert inputs that drive
both the rule-based scoring engine (``rules.py``) and the label
generation for ML training (``modeling.py``).  Changing a weight here
propagates through the entire pipeline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core CBC Junior School subjects (Grades 7–9).
# The order here determines column ordering in the feature matrix.
# ---------------------------------------------------------------------------
SUBJECTS = [
    "English",
    "Kiswahili or Kenyan Sign Language",
    "Mathematics",
    "Integrated Science",
    "Health Education",
    "Social Studies",
    "Religious Education",
    "Pre-Technical and Pre-Career Education",
    "Agriculture",
    "Business Studies",
    "Life Skills Education",
    "Sports and Physical Education",
]

# Machine-readable slug for each subject, used as column name prefixes
# in the feature matrix (e.g. ``mathematics_mean``).
SUBJECT_TO_SLUG = {
    "English": "english",
    "Kiswahili or Kenyan Sign Language": "kiswahili_or_kenyan_sign_language",
    "Mathematics": "mathematics",
    "Integrated Science": "integrated_science",
    "Health Education": "health_education",
    "Social Studies": "social_studies",
    "Religious Education": "religious_education",
    "Pre-Technical and Pre-Career Education": "pre_technical_and_pre_career_education",
    "Agriculture": "agriculture",
    "Business Studies": "business_studies",
    "Life Skills Education": "life_skills_education",
    "Sports and Physical Education": "sports_and_physical_education",
}

# Aliases allow CSV files to use shorthand subject names (e.g. "Kiswahili")
# which the ingestion layer normalises to the canonical form above.
SUBJECT_ALIASES = {
    "English": "English",
    "Kiswahili": "Kiswahili or Kenyan Sign Language",
    "Kenyan Sign Language": "Kiswahili or Kenyan Sign Language",
    "Kiswahili or Kenyan Sign Language": "Kiswahili or Kenyan Sign Language",
    "KSL": "Kiswahili or Kenyan Sign Language",
    "Mathematics": "Mathematics",
    "Integrated Science": "Integrated Science",
    "Health Education": "Health Education",
    "Social Studies": "Social Studies",
    "Religious Education": "Religious Education",
    "Pre-Technical and Pre-Career Education": "Pre-Technical and Pre-Career Education",
    "Agriculture": "Agriculture",
    "Business Studies": "Business Studies",
    "Life Skills Education": "Life Skills Education",
    "Sports and Physical Education": "Sports and Physical Education",
}

# ---------------------------------------------------------------------------
# Grade-term sequence — the chronological ordering of assessment windows
# used for trend calculation and feature engineering.
# ---------------------------------------------------------------------------
GRADE_TERM_SEQUENCE = [
    ("Grade 7", "Term 1"),
    ("Grade 7", "Term 2"),
    ("Grade 7", "Term 3"),
    ("Grade 8", "Term 1"),
    ("Grade 8", "Term 2"),
    ("Grade 8", "Term 3"),
    ("Grade 9", "Term 1"),
    ("Grade 9", "Term 2"),
    ("Grade 9", "Term 3"),
]

# Fast lookup to convert a (grade, term) pair into its chronological
# position (0–8), used by the feature builder for score ordering.
TERM_INDEX = {pair: index for index, pair in enumerate(GRADE_TERM_SEQUENCE)}

# ---------------------------------------------------------------------------
# Track and pathway configuration.
#
# Kenya's CBC Senior School has three pathways, each containing specific
# tracks.  The ``weights`` dictionary for each track defines how much
# each Junior School subject contributes to readiness for that track.
# Weights must sum to 1.0 per track.
# ---------------------------------------------------------------------------
TRACK_CONFIG = {
    "Pure Sciences": {
        "pathway": "STEM",
        "weights": {
            "English": 0.08,
            "Kiswahili or Kenyan Sign Language": 0.04,
            "Mathematics": 0.24,
            "Integrated Science": 0.27,
            "Health Education": 0.10,
            "Social Studies": 0.03,
            "Religious Education": 0.02,
            "Pre-Technical and Pre-Career Education": 0.10,
            "Agriculture": 0.06,
            "Business Studies": 0.02,
            "Life Skills Education": 0.01,
            "Sports and Physical Education": 0.03,
        },
    },
    "Applied Sciences": {
        "pathway": "STEM",
        "weights": {
            "English": 0.07,
            "Kiswahili or Kenyan Sign Language": 0.04,
            "Mathematics": 0.16,
            "Integrated Science": 0.22,
            "Health Education": 0.14,
            "Social Studies": 0.04,
            "Religious Education": 0.02,
            "Pre-Technical and Pre-Career Education": 0.09,
            "Agriculture": 0.13,
            "Business Studies": 0.04,
            "Life Skills Education": 0.02,
            "Sports and Physical Education": 0.03,
        },
    },
    "Technical Studies": {
        "pathway": "STEM",
        "weights": {
            "English": 0.07,
            "Kiswahili or Kenyan Sign Language": 0.03,
            "Mathematics": 0.22,
            "Integrated Science": 0.15,
            "Health Education": 0.06,
            "Social Studies": 0.03,
            "Religious Education": 0.01,
            "Pre-Technical and Pre-Career Education": 0.26,
            "Agriculture": 0.07,
            "Business Studies": 0.05,
            "Life Skills Education": 0.02,
            "Sports and Physical Education": 0.03,
        },
    },
    "Humanities and Business Studies": {
        "pathway": "Social Sciences",
        "weights": {
            "English": 0.18,
            "Kiswahili or Kenyan Sign Language": 0.16,
            "Mathematics": 0.08,
            "Integrated Science": 0.06,
            "Health Education": 0.04,
            "Social Studies": 0.18,
            "Religious Education": 0.12,
            "Pre-Technical and Pre-Career Education": 0.03,
            "Agriculture": 0.05,
            "Business Studies": 0.07,
            "Life Skills Education": 0.02,
            "Sports and Physical Education": 0.01,
        },
    },
    "Liberal Arts": {
        "pathway": "Arts and Sports Science",
        "weights": {
            "English": 0.18,
            "Kiswahili or Kenyan Sign Language": 0.14,
            "Mathematics": 0.04,
            "Integrated Science": 0.04,
            "Health Education": 0.05,
            "Social Studies": 0.14,
            "Religious Education": 0.10,
            "Pre-Technical and Pre-Career Education": 0.02,
            "Agriculture": 0.03,
            "Business Studies": 0.04,
            "Life Skills Education": 0.12,
            "Sports and Physical Education": 0.10,
        },
    },
    "Sports Science": {
        "pathway": "Arts and Sports Science",
        "weights": {
            "English": 0.10,
            "Kiswahili or Kenyan Sign Language": 0.08,
            "Mathematics": 0.08,
            "Integrated Science": 0.14,
            "Health Education": 0.14,
            "Social Studies": 0.08,
            "Religious Education": 0.04,
            "Pre-Technical and Pre-Career Education": 0.03,
            "Agriculture": 0.03,
            "Business Studies": 0.02,
            "Life Skills Education": 0.10,
            "Sports and Physical Education": 0.16,
        },
    },
}

# ---------------------------------------------------------------------------
# Derived lookup tables — built once from TRACK_CONFIG so the rest of the
# codebase can look up pathways, tracks, and their relationships cheaply.
# ---------------------------------------------------------------------------
PATHWAYS = sorted({details["pathway"] for details in TRACK_CONFIG.values()})
TRACKS = list(TRACK_CONFIG)
PATHWAY_TO_TRACKS = {
    pathway: [track for track, details in TRACK_CONFIG.items() if details["pathway"] == pathway]
    for pathway in PATHWAYS
}

# ---------------------------------------------------------------------------
# Schema and feature constants.
# ---------------------------------------------------------------------------

# Required columns in every assessment CSV or DataFrame.
CSV_COLUMNS = [
    "learner_id",
    "learner_name",
    "grade",
    "term",
    "subject",
    "score",
    "assessment_date",
]

# Statistical suffixes appended to each subject slug to form feature names
# (e.g. ``mathematics_mean``, ``mathematics_trend``).
FEATURE_SUFFIXES = ("mean", "recent_mean", "trend", "consistency")


def pathway_from_track(track: str) -> str:
    """Return the parent pathway name for a given track."""
    return TRACK_CONFIG[track]["pathway"]
