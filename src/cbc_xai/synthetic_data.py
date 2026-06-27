"""Synthetic assessment data generator for training and demonstration.

Produces realistic-looking CBC assessment records (score 35–95) for a
configurable number of learners across all nine grade-term slots and
twelve subjects.  Each learner is assigned a hidden "target track" that
biases their subject scores via the track weight configuration, creating
the correlations that the ML models learn to detect.

The generator is deterministic for a given ``random_seed``, ensuring
that the training pipeline and test suite produce reproducible results.
"""

from __future__ import annotations

import random
import itertools
from datetime import date

import pandas as pd

from .domain import CSV_COLUMNS, GRADE_TERM_SEQUENCE, SUBJECTS, TRACK_CONFIG, TRACKS

# Fixed assessment dates per grade-term slot.  These represent typical
# end-of-term assessment windows in the Kenyan school calendar.
TERM_DATES = {
    ("Grade 7", "Term 1"): date(2023, 4, 15),
    ("Grade 7", "Term 2"): date(2023, 8, 15),
    ("Grade 7", "Term 3"): date(2023, 11, 20),
    ("Grade 8", "Term 1"): date(2024, 4, 15),
    ("Grade 8", "Term 2"): date(2024, 8, 15),
    ("Grade 8", "Term 3"): date(2024, 11, 20),
    ("Grade 9", "Term 1"): date(2025, 4, 15),
    ("Grade 9", "Term 2"): date(2025, 8, 15),
    ("Grade 9", "Term 3"): date(2025, 11, 20),
}


def _base_subject_profile(track: str, general_strength: float) -> dict[str, float]:
    """Generate a learner's baseline subject scores biased toward *track*.

    Higher track weights produce higher baseline scores, simulating the
    natural aptitude pattern that the rule engine and models expect.
    """
    weights = TRACK_CONFIG[track]["weights"]
    return {
        subject: general_strength + (weights.get(subject, 0.0) * 35.0) + random.uniform(-6.0, 6.0)
        for subject in SUBJECTS
    }



# A broad selection of common English first names.
ENGLISH_FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Charles", "Joseph", "Thomas",
    "Christopher", "Daniel", "Paul", "Mark", "Donald", "George", "Kenneth", "Steven", "Edward", "Brian",
    "Ronald", "Anthony", "Kevin", "Jason", "Matthew", "Gary", "Timothy", "Jose", "Larry", "Jeffrey",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle",
    "Dorothy", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
    "Faith", "Joy", "Grace", "Hope", "Charity", "Mercy", "Victor", "Emmanuel", "Ian", "Brian", "Allan"
]

# A broad selection of common Kenyan surnames across various communities.
KENYAN_SURNAMES = [
    "Njoroge", "Kariuki", "Mutuku", "Ogot", "Wanjala", "Koech", "Mwema", "Otieno", "Ndung'u", "Macharia",
    "Kipkemboi", "Odhiambo", "Oloo", "Waweru", "Nyong'o", "Mboya", "Korir", "Cheruiyot", "Kenyatta", "Odinga",
    "Wafula", "Wanyonyi", "Sitienei", "Maina", "Wanjiku", "Ochieng", "Akinyi", "Kipkorir", "Chebet", "Mutua",
    "Mwikali", "Onyango", "Auma", "Kamau", "Njeri", "Atieno", "Kimutai", "Jepkemboi", "Mwangi", "Nyambura",
    "Kiplagat", "Wamalwa", "Naliaka", "Omondi", "Achieng", "Karanja", "Kiprop", "Kibaki", "Ruto", "Moi",
    "Gachagua", "Mudavadi", "Kalonzo", "Nkaissery", "Matiang'i", "Arap", "Kiptoo", "Kipchoge", "Wanyama", "Oliech"
]


def generate_synthetic_assessments(
    learner_count: int = 240,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic assessment records.

    Each learner receives 9 terms × 12 subjects = 108 rows.  Scores
    evolve over time using a per-learner improvement rate and per-subject
    volatility, producing realistic trend and consistency statistics for
    the feature builder.
    """
    random.seed(random_seed)
    records: list[dict[str, object]] = []

    # Generate all possible unique combinations of (First Name, Surname)
    all_possible_names = list(itertools.product(ENGLISH_FIRST_NAMES, KENYAN_SURNAMES))

    # Ensure we don't try to sample more names than we have combinations
    if learner_count > len(all_possible_names):
        raise ValueError(f"Cannot generate {learner_count} unique names from the provided pools.")

    # Sample exactly 'learner_count' unique names, guaranteeing no exact duplicates
    selected_names = random.sample(all_possible_names, learner_count)

    for learner_number in range(1, learner_count + 1):
        learner_id = f"L{learner_number:03d}"
        first_name, surname = selected_names[learner_number - 1]
        learner_name = f"{first_name} {surname}"
        target_track = random.choice(TRACKS)
        general_strength = random.uniform(48.0, 78.0)
        overall_improvement = random.uniform(-0.4, 1.4)
        subject_bases = _base_subject_profile(target_track, general_strength)

        for term_index, (grade, term) in enumerate(GRADE_TERM_SEQUENCE):
            for subject in SUBJECTS:
                # Learners improve faster in subjects that are heavily
                # weighted for their target track, reinforcing the signal.
                track_weight = TRACK_CONFIG[target_track]["weights"].get(subject, 0.0)
                trend_multiplier = 1.0 + (track_weight * 1.7)
                trend_component = overall_improvement * trend_multiplier * term_index
                volatility = random.uniform(-5.5, 5.5)
                score = subject_bases[subject] + trend_component + volatility
                score = max(35.0, min(95.0, score))
                records.append(
                    {
                        "learner_id": learner_id,
                        "learner_name": learner_name,
                        "grade": grade,
                        "term": term,
                        "subject": subject,
                        "score": round(score, 2),
                        "assessment_date": TERM_DATES[(grade, term)].isoformat(),
                    }
                )

    return pd.DataFrame(records, columns=CSV_COLUMNS)
