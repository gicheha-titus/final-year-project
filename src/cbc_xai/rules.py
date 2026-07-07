"""Rule-based readiness scoring and guidance note generation.

This module implements the domain-expert scoring logic that converts a
learner's per-subject statistical summary into:

1. **Track scores** — a weighted composite for each of the six tracks,
   with bonus/penalty adjustments for subject-specific thresholds.
2. **Probability distributions** — softmax-style normalisation of the
   raw scores into pathway and track probabilities.
3. **Guidance notes** — narrative suggestions for teacher–parent
   conversations based on subject strengths and trends.


"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

from .domain import PATHWAY_TO_TRACKS, PATHWAYS, SUBJECTS, TRACK_CONFIG, pathway_from_track


def _clamp(score: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Constrain *score* to [minimum, maximum]."""
    return max(minimum, min(maximum, score))


def track_score_from_summary(
    track: str, subject_summary: Mapping[str, Mapping[str, float]]
) -> float:
    """Compute a weighted readiness score for one track.

    The base score combines long-run mean (55%), recent mean (25%),
    trend (scaled ×2), and consistency (penalised at 20%).  Track-specific
    bonus and penalty rules then adjust the score to reflect curriculum
    domain knowledge — e.g. Pure Sciences requires strong Mathematics
    *and* Integrated Science recent performance.
    """
    weights = TRACK_CONFIG[track]["weights"]
    weighted_mean = sum(
        subject_summary[subject]["mean"] * weights.get(subject, 0.0) for subject in SUBJECTS
    )
    weighted_recent = sum(
        subject_summary[subject]["recent_mean"] * weights.get(subject, 0.0) for subject in SUBJECTS
    )
    weighted_trend = sum(
        subject_summary[subject]["trend"] * weights.get(subject, 0.0) for subject in SUBJECTS
    )
    weighted_consistency = sum(
        subject_summary[subject]["consistency"] * weights.get(subject, 0.0) for subject in SUBJECTS
    )

    # Composite formula: stability and recent trajectory matter more than
    # historical average alone, which is why trend is amplified (×2) and
    # consistency is penalised (higher std → lower score).
    raw_score = (
        (0.55 * weighted_mean)
        + (0.25 * weighted_recent)
        + (2.0 * weighted_trend)
        - (0.20 * weighted_consistency)
    )

    # ---- Track-specific bonus / penalty rules ----
    # These thresholds encode domain knowledge about prerequisite strength
    # for each track.  They intentionally differ between tracks because the
    # curriculum expectations vary.
    if track == "Pure Sciences":
        if (
            subject_summary["Mathematics"]["recent_mean"] > 70
            and subject_summary["Integrated Science"]["recent_mean"] > 70
            and subject_summary["Health Education"]["recent_mean"] > 65
        ):
            raw_score += 7.0
        if (
            min(
                subject_summary["Mathematics"]["mean"],
                subject_summary["Integrated Science"]["mean"],
            )
            < 52
        ):
            raw_score -= 8.0
    elif track == "Applied Sciences":
        if (
            subject_summary["Agriculture"]["recent_mean"] > 68
            and subject_summary["Integrated Science"]["recent_mean"] > 66
            and subject_summary["Health Education"]["recent_mean"] > 60
        ):
            raw_score += 6.0
        if subject_summary["Agriculture"]["trend"] < 0:
            raw_score -= 3.5
    elif track == "Technical Studies":
        if (
            subject_summary["Pre-Technical and Pre-Career Education"]["recent_mean"] > 68
            and subject_summary["Mathematics"]["recent_mean"] > 64
        ):
            raw_score += 7.0
        if subject_summary["Pre-Technical and Pre-Career Education"]["mean"] < 55:
            raw_score -= 8.0
    elif track == "Humanities and Business Studies":
        if (
            subject_summary["English"]["recent_mean"] > 68
            and subject_summary["Kiswahili or Kenyan Sign Language"]["recent_mean"] > 68
            and subject_summary["Social Studies"]["recent_mean"] > 68
            and subject_summary["Religious Education"]["recent_mean"] > 60
        ):
            raw_score += 7.0
        if (
            subject_summary["English"]["trend"] < 0
            and subject_summary["Business Studies"]["trend"] < 0
        ):
            raw_score -= 4.0
    elif track == "Liberal Arts":
        if (
            subject_summary["Life Skills Education"]["recent_mean"] > 70
            and subject_summary["English"]["mean"] > 65
            and subject_summary["Social Studies"]["mean"] > 65
        ):
            raw_score += 6.5
        if subject_summary["Life Skills Education"]["consistency"] > 12:
            raw_score -= 4.0
    elif track == "Sports Science":
        if (
            subject_summary["Sports and Physical Education"]["recent_mean"] > 74
            and subject_summary["Health Education"]["recent_mean"] > 68
        ):
            raw_score += 6.0
        if subject_summary["Sports and Physical Education"]["trend"] < 0:
            raw_score -= 3.5

    return round(_clamp(raw_score), 4)


def compute_track_scores(subject_summary: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Return a score for every track given a learner's subject summary."""
    return {track: track_score_from_summary(track, subject_summary) for track in TRACK_CONFIG}


def probability_distribution_from_scores(score_map: Mapping[str, float]) -> dict[str, float]:
    """Convert raw track scores into a softmax-style probability distribution.

    Uses a temperature of 7.5 and a baseline shift to the maximum score
    for numerical stability.  The result sums to 1.0 and can be
    interpreted as "readiness confidence" across tracks.
    """
    if not score_map:
        return {}
    baseline = max(score_map.values())
    exponentials = {label: math.exp((score - baseline) / 7.5) for label, score in score_map.items()}
    total = sum(exponentials.values()) or 1.0
    return {label: round(value / total, 6) for label, value in exponentials.items()}


def pathway_probabilities_from_track_probabilities(
    track_probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Aggregate track-level probabilities up to the pathway level.

    Each pathway's probability is the sum of its constituent tracks'
    probabilities, reflecting the total readiness evidence for the
    broader pathway family.
    """
    grouped: dict[str, float] = defaultdict(float)
    for track, probability in track_probabilities.items():
        grouped[pathway_from_track(track)] += probability
    return {pathway: round(grouped.get(pathway, 0.0), 6) for pathway in PATHWAYS}


def derive_labels(subject_summary: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    """Derive the full set of readiness labels for a learner.

    This is the main entry point used during model training to generate
    ground-truth labels and during prediction to produce the rule-based
    component of the recommendation.
    """
    track_scores = compute_track_scores(subject_summary)
    track_probabilities = probability_distribution_from_scores(track_scores)
    pathway_probabilities = pathway_probabilities_from_track_probabilities(track_probabilities)
    best_track = max(track_scores, key=track_scores.get)
    return {
        "track_scores": track_scores,
        "track_probabilities": track_probabilities,
        "pathway_probabilities": pathway_probabilities,
        "predicted_track": best_track,
        "predicted_pathway": pathway_from_track(best_track),
        "top_track_score": track_scores[best_track],
    }


def build_guidance_notes(
    subject_summary: Mapping[str, Mapping[str, float]],
    predicted_pathway: str,
) -> list[str]:
    """Generate actionable narrative guidance for the predicted pathway.

    Focuses on the top three most-weighted subjects for the pathway and
    produces a note for each based on the learner's recent performance
    and trend direction.  These notes are included in the PDF report and
    the application UI.
    """
    pathway_tracks = PATHWAY_TO_TRACKS[predicted_pathway]

    # Accumulate subject weights across all tracks in the pathway so
    # that the most important subjects are addressed first.
    pathway_weights = defaultdict(float)
    for track in pathway_tracks:
        for subject, weight in TRACK_CONFIG[track]["weights"].items():
            pathway_weights[subject] += weight

    ranked_subjects = sorted(
        pathway_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    notes: list[str] = []
    for subject, _ in ranked_subjects[:3]:
        recent = subject_summary[subject]["recent_mean"]
        trend = subject_summary[subject]["trend"]
        if recent < 60:
            notes.append(
                f"Strengthen {subject} through focused remediation because recent scores are below the pathway comfort zone."
            )
        elif trend < 0:
            notes.append(
                f"Monitor {subject} closely because the recent trend is declining even though it remains important for {predicted_pathway}."
            )
        else:
            notes.append(
                f"Maintain the current trajectory in {subject}; it is supporting readiness for {predicted_pathway}."
            )
    return notes
