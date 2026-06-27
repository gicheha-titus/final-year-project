from __future__ import annotations

from cbc_xai.rules import derive_labels


def test_rule_engine_returns_track_and_pathway() -> None:
    subject_summary = {
        "English": {"mean": 70, "recent_mean": 72, "trend": 0.4, "consistency": 6},
        "Kiswahili or Kenyan Sign Language": {"mean": 68, "recent_mean": 70, "trend": 0.3, "consistency": 6},
        "Mathematics": {"mean": 80, "recent_mean": 83, "trend": 1.2, "consistency": 4},
        "Integrated Science": {"mean": 82, "recent_mean": 84, "trend": 1.1, "consistency": 5},
        "Health Education": {"mean": 74, "recent_mean": 76, "trend": 0.8, "consistency": 5},
        "Social Studies": {"mean": 60, "recent_mean": 61, "trend": 0.1, "consistency": 7},
        "Religious Education": {"mean": 62, "recent_mean": 64, "trend": 0.2, "consistency": 7},
        "Pre-Technical and Pre-Career Education": {"mean": 76, "recent_mean": 78, "trend": 0.9, "consistency": 5},
        "Agriculture": {"mean": 64, "recent_mean": 66, "trend": 0.5, "consistency": 6},
        "Business Studies": {"mean": 58, "recent_mean": 60, "trend": 0.2, "consistency": 6},
        "Life Skills Education": {"mean": 61, "recent_mean": 63, "trend": 0.3, "consistency": 6},
        "Sports and Physical Education": {"mean": 57, "recent_mean": 59, "trend": 0.2, "consistency": 7},
    }
    labels = derive_labels(subject_summary)
    assert labels["predicted_pathway"] == "STEM"
    assert labels["predicted_track"] in {"Pure Sciences", "Technical Studies", "Applied Sciences"}
