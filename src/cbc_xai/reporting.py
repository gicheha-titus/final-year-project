"""PDF report generation for learner readiness summaries.

Produces a single-page A4 PDF containing the learner's recommended
pathway, track probabilities, supporting and limiting subject factors,
and actionable guidance notes.  The report is designed for
parent–teacher guidance conversations — not automated placement — so
the layout prioritises readability and transparency over compactness.

Advisory framing note: every surface of this report must reinforce that
the recommendation is a decision-support signal, not a placement decision.
The disclaimer at the bottom of the report is not boilerplate — it reflects
the actual limitation of a model trained on synthetic data.
"""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import REPORTS_DIR, ensure_directories, readiness_band
from .exceptions import ReportError
from .modeling import PredictionOutput

log = logging.getLogger(__name__)


def _probability_table_rows(probabilities: dict[str, float]) -> list[list[str]]:
    """Format a probability dictionary into table rows sorted descending."""
    rows = [["Option", "Probability", "Readiness"]]
    rows.extend(
        [
            [label, f"{value * 100:.1f}%", readiness_band(value)]
            for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        ]
    )
    return rows


def generate_pdf_report(prediction: PredictionOutput) -> Path:
    """Generate and save a readiness report PDF for a single learner.

    Returns the absolute path to the saved PDF so that the UI can
    display a confirmation and the storage layer can log the report.

    Raises ``ReportError`` if PDF generation fails.
    """
    ensure_directories()
    report_path = REPORTS_DIR / f"{prediction.learner_id}_readiness_report.pdf"

    try:
        document = SimpleDocTemplate(str(report_path), pagesize=A4)
        styles = getSampleStyleSheet()
        dominant_probability = max(prediction.pathway_probabilities.values())

        story = [
            Paragraph("CBC Pathway and Track Readiness Guidance Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Learner ID: {prediction.learner_id}", styles["Normal"]),
            Paragraph(f"Learner Name: {prediction.learner_name}", styles["Normal"]),
            Paragraph(f"Recommended Pathway: {prediction.predicted_pathway}", styles["Normal"]),
            Paragraph(f"Top Track: {prediction.predicted_track}", styles["Normal"]),
            Paragraph(
                f"Readiness: {readiness_band(dominant_probability)} "
                f"({dominant_probability * 100:.1f}%)",
                styles["Normal"],
            ),
            Spacer(1, 12),
            Paragraph("Pathway Readiness", styles["Heading2"]),
        ]

        pathway_table = Table(_probability_table_rows(prediction.pathway_probabilities), hAlign="LEFT")
        pathway_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D8E8FF")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([pathway_table, Spacer(1, 12), Paragraph("Track Readiness", styles["Heading2"])])

        track_table = Table(_probability_table_rows(prediction.track_probabilities), hAlign="LEFT")
        track_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4F5E8")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([track_table, Spacer(1, 12), Paragraph("Supporting Factors", styles["Heading2"])])

        for item in prediction.strengths:
            story.append(Paragraph(f"- {item['subject']}: currently supporting this recommendation.", styles["Normal"]))
        story.extend([Spacer(1, 8), Paragraph("Watch Areas", styles["Heading2"])])
        for item in prediction.limiting_factors:
            story.append(Paragraph(f"- {item['subject']}: may need additional support.", styles["Normal"]))
        story.extend([Spacer(1, 8), Paragraph("Guidance Notes", styles["Heading2"])])
        for note in prediction.guidance_notes:
            story.append(Paragraph(f"- {note}", styles["Normal"]))

        # Advisory disclaimer — this framing is a research requirement, not decoration.
        story.extend([
            Spacer(1, 20),
            Paragraph(
                "<b>Important:</b> This report is a decision-support tool for teacher–parent "
                "guidance conversations. It is not an automated placement decision. "
                "Recommendations are based on assessment patterns from this system and "
                "should be considered alongside teacher judgment and learner context.",
                styles["Normal"],
            ),
        ])

        document.build(story)
    except Exception as exc:
        raise ReportError(f"Failed to generate PDF for {prediction.learner_id}: {exc}") from exc

    log.info("Report generated at %s", report_path)
    return report_path
