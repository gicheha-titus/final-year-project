"""PySide6 desktop application for CBC pathway readiness guidance.

This module contains the complete UI layer: login dialog, sidebar
navigation, and six workspace pages (Dashboard, Data Intake, Learner
Workspace, Cohort Insights, Report Center, User Accounts).

Architecture notes:

- ``main()`` runs a session loop: LoginDialog → MainWindow → repeat or
  quit.  This allows account switching without restarting the process.
- ``MainWindow`` holds all page widgets in a ``QStackedWidget`` and
  owns the current prediction state, assessment cache, and model bundle.
- Heavy operations (model prediction, SHAP) run synchronously because
  the prototype targets single-user offline desktops, not concurrent
  server workloads.
- Logging goes to the rotating file handler configured in
  ``logging_config.py``; the UI shows plain-language messages only.
"""

from __future__ import annotations

import logging
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import MODEL_BUNDLE_PATH, SYNTHETIC_ASSESSMENTS_CSV, readiness_band
from .domain import pathway_from_track
from .exceptions import IngestionError, ReportError
from .features import build_feature_matrix, build_subject_summary
from .ingestion import load_assessment_csv
from .logging_config import configure_logging
from .modeling import (
    PredictionOutput,
    load_model_bundle,
    predict_for_learner,
    train_and_select_model,
)
from .reporting import generate_pdf_report
from .storage import (
    authenticate_user,
    backup_database,
    count_reports,
    create_user,
    delete_user,
    import_assessment_frame,
    initialize_database,
    latest_report_path,
    list_users,
    load_all_assessments,
    load_assessments_for_learner,
    load_learners,
    reset_user_password,
    save_prediction,
    save_report_record,
    set_user_active,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global stylesheet — uses Qt's QSS (CSS-like) syntax.
# The colour palette follows a warm neutral base (#f1ece4) with a dark
# navy sidebar (#0f2238) and gold accent (#d79b4c).  Every widget type
# is styled here so that individual page builders never embed ad-hoc
# colours, keeping the visual identity consistent.
# ---------------------------------------------------------------------------
APP_STYLESHEET = """
QWidget {
    color: #11283e;
    font-family: "Candara";
    font-size: 14px;
}
QLabel {
    background: transparent;
    background-color: transparent;
    border: none;
    padding: 0px;
}
QMainWindow, QDialog {
    background: #f1ece4;
}
QFrame#sidebar {
    background: #0f2238;
    border-radius: 28px;
}
QFrame#contentShell {
    background: transparent;
}
QFrame#pageHeader,
QFrame#sectionCard,
QFrame#statCard,
QFrame#panelCard,
QFrame#reportBanner,
QFrame#loginPanel {
    background: #fffdf9;
    border: 1px solid #d3c5b4;
    border-radius: 24px;
}
QFrame#sidebarUserCard {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
}
QFrame#heroCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #10243d,
        stop: 0.55 #1d4758,
        stop: 1 #275e61
    );
    border-radius: 28px;
    border: none;
}
QFrame#loginBrand {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #17324c,
        stop: 0.55 #214965,
        stop: 1 #96652c
    );
    border-radius: 28px;
    border: none;
}
QLabel#brandTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
    font-family: "Georgia";
    font-size: 30px;
    font-weight: 700;
}
QLabel#brandSubtitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f6f8fb;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.4;
}
QLabel#brandBullet {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f8fafc;
    font-size: 14px;
    font-weight: 600;
}
QLabel#pageKicker,
QLabel#sectionEyebrow {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #7b5127;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#pageTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #10243d;
    font-family: "Georgia";
    font-size: 28px;
    font-weight: 700;
}
QLabel#pageSubtitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #31495c;
    font-size: 14px;
}
QLabel#sectionTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #10243d;
    font-family: "Georgia";
    font-size: 21px;
    font-weight: 700;
}
QLabel#sectionDescription {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #334c5e;
    font-size: 13px;
}
QLabel#heroTitle,
QLabel#heroValue {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
}
QLabel#heroTitle {
    font-family: "Georgia";
    font-size: 29px;
    font-weight: 700;
}
QLabel#heroDescription {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f0f7fb;
    font-size: 14px;
}
QLabel#heroMetricLabel {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f3d8ac;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#heroMetricValue {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #fffaf3;
    font-size: 24px;
    font-weight: 700;
}
QLabel#statTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #3c5567;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#statValue {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #10243d;
    font-size: 26px;
    font-weight: 700;
}
QLabel#statDetail {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #344d60;
    font-size: 13px;
}
QLabel#sidebarHeadline {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
    font-family: "Georgia";
    font-size: 24px;
    font-weight: 700;
}
QLabel#sidebarCaption {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f0f7fb;
    font-size: 13px;
    font-weight: 600;
}
QLabel#userCardTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f3d8ac;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#userCardValue {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
}
QFrame#sidebarUserCard QLabel#userCardTitle {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f3d8ac;
}
QFrame#sidebarUserCard QLabel#userCardValue {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
}
QFrame#sidebarUserCard QLabel#sidebarCaption {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #f0f7fb;
    font-weight: 600;
}
QPushButton#navButton {
    background: transparent;
    color: #dbe3ea;
    border: none;
    border-radius: 18px;
    padding: 14px 18px;
    text-align: left;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#navButton:hover {
    background: rgba(255, 255, 255, 0.12);
}
QPushButton#navButton:checked {
    background: #d79b4c;
    color: #10243d;
}
QPushButton {
    border-radius: 16px;
    border: 1px solid #c8baa8;
    padding: 11px 18px;
    background: #f7f1e8;
    color: #17324c;
    font-weight: 700;
}
QPushButton:hover {
    background: #ecdfcc;
}
QPushButton:disabled {
    color: #6f7980;
    background: #e9e1d5;
}
QPushButton[variant="primary"] {
    background: #d79b4c;
    color: #10243d;
    border: none;
}
QPushButton[variant="primary"]:hover {
    background: #e3aa60;
}
QPushButton[variant="ghost"] {
    background: rgba(255, 250, 243, 0.18);
    color: #fffaf3;
    border: 1px solid rgba(255, 250, 243, 0.34);
}
QPushButton[variant="ghost"]:hover {
    background: rgba(255, 250, 243, 0.28);
}
QLineEdit, QComboBox, QTextBrowser, QTableWidget {
    background: #ffffff;
    border: 1px solid #cdbfae;
    border-radius: 16px;
    padding: 10px 12px;
    color: #17324c;
}
QLineEdit:focus, QComboBox:focus, QTextBrowser:focus, QTableWidget:focus {
    border: 1px solid #1d5b63;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QTableWidget {
    gridline-color: #e1d7cb;
    selection-background-color: #d8e8ea;
    alternate-background-color: #fbf6ef;
}
QHeaderView::section {
    background: #ecdfce;
    color: #17324c;
    border: none;
    border-bottom: 1px solid #ccbead;
    padding: 10px;
    font-weight: 700;
}
QTextBrowser {
    line-height: 1.5;
}
QLabel#statusBanner {
    border-radius: 16px;
    padding: 12px 14px;
    font-weight: 700;
}
QLabel#statusBanner[tone="info"] {
    background: #e7f0f8;
    color: #173f5b;
}
QLabel#statusBanner[tone="success"] {
    background: #e4f3ea;
    color: #1f4e3a;
}
QLabel#statusBanner[tone="error"] {
    background: #fde9e6;
    color: #7a281f;
}
QLabel#badge {
    background: #e7dbc8;
    color: #3f2d1b;
    border: 1px solid #cdbba4;
    border-radius: 14px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 700;
}
"""

# Human-readable title and description for each page, used in the
# header bar when the user navigates between workspace sections.
PAGE_METADATA = {
    "dashboard": (
        "Home",
        "View learner activity, report availability, and pathway distribution across the current cohort.",
    ),
    "import": (
        "Data Intake",
        "Validate assessment files before import and keep the workflow reliable for school staff.",
    ),
    "learner": (
        "Learner Workspace",
        "Review one learner at a time with the recommendation, evidence, and guidance on a single screen.",
    ),
    "cohort": (
        "Cohort Insights",
        "Observe whole-group pathway patterns to support intervention planning and school-level guidance.",
    ),
    "report": (
        "Report Center",
        "Prepare a parent-facing readiness summary and export the current learner report as PDF.",
    ),
    "users": (
        "User Accounts",
        "Create additional teacher accounts and review who can access the workspace.",
    ),
}

# Colour palettes for pathway and track visualisations.  These match
# the chart bars, probability bars, and subject table highlights.
PATHWAY_COLORS = {
    "STEM": "#2d6d68",
    "Social Sciences": "#d79b4c",
    "Arts and Sports Science": "#b45745",
}

TRACK_COLORS = {
    "Pure Sciences": "#2d6d68",
    "Applied Sciences": "#3f8a74",
    "Technical Studies": "#2c4d73",
    "Humanities and Business Studies": "#d79b4c",
    "Liberal Arts": "#be6c4d",
    "Sports Science": "#b45745",
}


def ensure_seed_data() -> None:
    """Guarantee that the database and model bundle exist before the UI starts.

    On a fresh install this triggers synthetic data generation and model
    training so the workspace is immediately usable with sample data.
    """
    initialize_database()
    current_assessments = load_all_assessments()
    if MODEL_BUNDLE_PATH.exists() and not current_assessments.empty:
        return

    results = train_and_select_model()
    if current_assessments.empty:
        import_assessment_frame(results["assessments"])
    elif not SYNTHETIC_ASSESSMENTS_CSV.exists():
        results["assessments"].to_csv(SYNTHETIC_ASSESSMENTS_CSV, index=False)


def apply_shadow(widget: QWidget, blur: int = 32, y_offset: int = 8) -> None:
    """Attach a subtle drop shadow to *widget* for depth on card elements."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y_offset)
    shadow.setColor(QColor(16, 36, 61, 28))
    widget.setGraphicsEffect(shadow)


def repolish(widget: QWidget) -> None:
    """Force Qt to re-evaluate dynamic properties (e.g. tone) on *widget*."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_banner_state(widget: QLabel, text: str, tone: str) -> None:
    """Update a status banner's text and visual tone (info/success/error)."""
    widget.setText(text)
    widget.setProperty("tone", tone)
    repolish(widget)


def clear_layout(layout: QVBoxLayout | QHBoxLayout | QGridLayout) -> None:
    """Recursively remove all child widgets from *layout*."""
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        child_layout = item.layout()
        if child is not None:
            child.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


def create_scroll_page(page: QWidget) -> QScrollArea:
    """Wrap *page* in a frameless scroll area for overflow handling."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(page)
    return scroll


def trend_label(trend: float) -> tuple[str, str]:
    """Return a human-readable label and colour for a trend slope value."""
    if trend >= 0.35:
        return "Improving", "#2d6d68"
    if trend <= -0.35:
        return "Needs attention", "#b45745"
    return "Stable", "#d79b4c"


class SectionCard(QFrame):
    """Reusable card container with optional eyebrow, title, and body layout."""

    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        eyebrow: str = "",
        accent: bool = False,
        object_name: str | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName(object_name or ("heroCard" if accent else "sectionCard"))
        apply_shadow(self, blur=26, y_offset=10 if accent else 8)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(14)

        if eyebrow:
            eyebrow_label = QLabel(eyebrow)
            eyebrow_label.setObjectName("sectionEyebrow")
            self.layout.addWidget(eyebrow_label)
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("heroTitle" if accent else "sectionTitle")
            title_label.setWordWrap(True)
            self.layout.addWidget(title_label)
        if description:
            description_label = QLabel(description)
            description_label.setObjectName("heroDescription" if accent else "sectionDescription")
            description_label.setWordWrap(True)
            self.layout.addWidget(description_label)

        self.body = QVBoxLayout()
        self.body.setSpacing(14)
        self.layout.addLayout(self.body)


class StatCard(QFrame):
    """Compact metric card showing a title, large value, and detail text."""

    def __init__(self, title: str, detail: str = "") -> None:
        super().__init__()
        self.setObjectName("statCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        apply_shadow(self, blur=24, y_offset=8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("statTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        self.value_label = QLabel("-")
        self.value_label.setObjectName("statValue")
        layout.addWidget(self.value_label)

        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("statDetail")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

    def set_content(self, value: str, detail: str = "") -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class ProbabilityBar(QFrame):
    """Horizontal progress bar widget for displaying pathway/track probabilities."""

    def __init__(self, accent: str) -> None:
        super().__init__()
        self.setObjectName("panelCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        self.title_label = QLabel("-")
        self.title_label.setStyleSheet("font-weight: 700; color: #17324c;")
        self.value_label = QLabel("0.0%")
        self.value_label.setStyleSheet("font-weight: 700; color: #3f5668;")
        top_row.addWidget(self.title_label)
        top_row.addStretch(1)
        top_row.addWidget(self.value_label)
        layout.addLayout(top_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            f"""
            QProgressBar {{
                background: #e2d6c7;
                border: none;
                border-radius: 9px;
                min-height: 12px;
                max-height: 12px;
            }}
            QProgressBar::chunk {{
                background: {accent};
                border-radius: 9px;
            }}
            """
        )
        layout.addWidget(self.progress)

        self.note_label = QLabel("")
        self.note_label.setStyleSheet("color: #42586a; font-size: 12px;")
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

    def set_content(self, label: str, probability: float, note: str = "") -> None:
        self.title_label.setText(label)
        self.value_label.setText(f"{probability * 100:.1f}%")
        self.progress.setValue(max(0, min(100, int(round(probability * 100)))))
        self.note_label.setText(note)


class LoginDialog(QDialog):
    """Modal login dialog with branded side panel and credential form."""

    def __init__(self) -> None:
        super().__init__()
        self.user: dict[str, str] | None = None
        self.setWindowTitle("CBC Pathway Guidance Login")
        self.setModal(True)
        self.resize(940, 560)

        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        brand = QFrame()
        brand.setObjectName("loginBrand")
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(34, 34, 34, 34)
        brand_layout.setSpacing(16)

        brand_title = QLabel("CBC Pathway Guidance")
        brand_title.setObjectName("brandTitle")
        brand_layout.addWidget(brand_title)

        brand_subtitle = QLabel(
            "A desktop decision-support workspace for explainable pathway readiness in Grades 7 to 9."
        )
        brand_subtitle.setObjectName("brandSubtitle")
        brand_subtitle.setWordWrap(True)
        brand_layout.addWidget(brand_subtitle)

        for bullet in (
            "Offline-first workflow for school environments.",
            "Subject-level explanations designed for teacher trust.",
            "Printable readiness reports for guidance conversations.",
        ):
            label = QLabel(f"- {bullet}")
            label.setObjectName("brandBullet")
            label.setWordWrap(True)
            brand_layout.addWidget(label)

        brand_layout.addStretch(1)
        credentials_note = QLabel("Default accounts:\nadmin / Admin@123\nteacher / Teacher@123")
        credentials_note.setObjectName("brandBullet")
        brand_layout.addWidget(credentials_note)

        panel = QFrame()
        panel.setObjectName("loginPanel")
        apply_shadow(panel, blur=28, y_offset=10)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(30, 30, 30, 30)
        panel_layout.setSpacing(16)

        eyebrow = QLabel("WELCOME")
        eyebrow.setObjectName("sectionEyebrow")
        panel_layout.addWidget(eyebrow)

        heading = QLabel("Sign in to the school guidance workspace")
        heading.setObjectName("sectionTitle")
        heading.setWordWrap(True)
        panel_layout.addWidget(heading)

        guidance = QLabel(
            "Use a local role account to import assessments, review recommendations, and export learner reports."
        )
        guidance.setObjectName("sectionDescription")
        guidance.setWordWrap(True)
        panel_layout.addWidget(guidance)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.handle_login)
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        panel_layout.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setObjectName("statusBanner")
        self.status_label.setWordWrap(True)
        set_banner_state(
            self.status_label,
            "Sign in with school role account.",
            "info",
        )
        panel_layout.addWidget(self.status_label)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        login_button = QPushButton("Enter Workspace")
        login_button.setProperty("variant", "primary")
        login_button.clicked.connect(self.handle_login)
        action_row.addWidget(login_button)
        panel_layout.addLayout(action_row)
        panel_layout.addStretch(1)

        root.addWidget(brand, 3)
        root.addWidget(panel, 2)

    def handle_login(self) -> None:
        user = authenticate_user(self.username_input.text().strip(), self.password_input.text())
        if not user:
            set_banner_state(
                self.status_label,
                "The credentials were not recognised. Try the default admin or teacher account.",
                "error",
            )
            return
        self.user = user
        self.accept()


class CohortCanvas(FigureCanvas):
    """Matplotlib canvas for the horizontal bar chart of pathway distribution."""

    def __init__(self) -> None:
        figure = Figure(figsize=(6, 4), tight_layout=True, facecolor="#fff9f1")
        self.axes = figure.add_subplot(111)
        self.axes.set_facecolor("#fff9f1")
        super().__init__(figure)

    def render_counts(self, counts: Counter[str]) -> None:
        self.axes.clear()
        self.figure.patch.set_facecolor("#fff9f1")
        self.axes.set_facecolor("#fff9f1")

        if not counts:
            self.axes.text(
                0.5,
                0.5,
                "Import learner data to see pathway distribution.",
                ha="center",
                va="center",
                fontsize=12,
                color="#5f7382",
            )
            self.axes.axis("off")
            self.draw()
            return

        labels = [label for label in PATHWAY_COLORS if label in counts]
        values = [counts[label] for label in labels]
        colors = [PATHWAY_COLORS.get(label, "#2d6d68") for label in labels]

        bars = self.axes.barh(labels, values, color=colors, height=0.58)
        self.axes.set_title("Cohort pathway distribution", fontsize=13, color="#17324c", pad=14)
        self.axes.tick_params(axis="y", colors="#17324c", labelsize=11)
        self.axes.tick_params(axis="x", colors="#6b7f8b", labelsize=10)
        self.axes.grid(axis="x", color="#e8ddd0", linestyle="--", linewidth=0.8)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["left"].set_visible(False)
        self.axes.spines["bottom"].set_color("#d7cab8")
        for bar, value in zip(bars, values, strict=True):
            self.axes.text(
                value + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                color="#17324c",
                fontsize=11,
                fontweight="bold",
            )
        self.draw()


class MainWindow(QMainWindow):
    """Primary workspace window containing all six application pages.

    Emits ``session_finished(restart: bool)`` on close so the session
    loop in ``main()`` can either restart with a new login or quit.
    """

    session_finished = Signal(bool)

    def __init__(self, user: dict[str, str]) -> None:
        super().__init__()
        self.user = user
        self._restart_session = False
        self.current_prediction: PredictionOutput | None = None
        self.current_learner_id: str | None = None
        self.current_subject_summary: dict[str, dict[str, float]] = {}
        self.import_preview_frame: pd.DataFrame | None = None
        self.import_preview_source: str | None = None
        self.all_assessments = pd.DataFrame()
        self.learner_records: list[dict[str, str]] = []
        self.user_records: list[dict[str, object]] = []
        self.cohort_counts: Counter[str] = Counter()
        self.latest_saved_report: str | None = None
        self.bundle = load_model_bundle()

        self.setWindowTitle("CBC Explainable Pathway Guidance")
        self.resize(1480, 920)
        self.setMinimumSize(1320, 840)

        self.nav_buttons: dict[str, QPushButton] = {}
        self.page_order = self._page_keys_for_user(self.user)

        self._build_ui()
        self.set_active_page("dashboard")
        self.refresh_all_views()

    def _page_keys_for_user(self, user: dict[str, str]) -> list[str]:
        keys = ["dashboard", "import", "learner", "cohort", "report"]
        if user.get("role") == "Admin":
            keys.append("users")
        return keys

    def switch_account(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Switch account",
                "Do you want to sign out of the current account and return to the login screen?",
            )
            != QMessageBox.Yes
        ):
            return
        self._restart_session = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        restart_session = self._restart_session
        self._restart_session = False
        event.accept()
        QTimer.singleShot(0, lambda restart=restart_session: self.session_finished.emit(restart))

    def _build_ui(self) -> None:
        shell = QWidget()
        root = QHBoxLayout(shell)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        sidebar = self._build_sidebar()
        content_shell = QFrame()
        content_shell.setObjectName("contentShell")
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        header = self._build_header()
        self.page_stack = QStackedWidget()
        self._build_pages()

        content_layout.addWidget(header)
        content_layout.addWidget(self.page_stack, 1)

        root.addWidget(sidebar)
        root.addWidget(content_shell, 1)
        self.setCentralWidget(shell)
        self.statusBar().showMessage("Workspace ready.")

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(270)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(22, 24, 22, 24)
        layout.setSpacing(12)

        title = QLabel("Workspace")
        title.setObjectName("sidebarHeadline")
        layout.addWidget(title)

        caption = QLabel(
            "Offline student pathway support with clear learner evidence and printable reports."
        )
        caption.setObjectName("sidebarCaption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        layout.addSpacing(18)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for key in self.page_order:
            title_text, _ = PAGE_METADATA[key]
            button = QPushButton(title_text)
            button.setCheckable(True)
            button.setObjectName("navButton")
            button.clicked.connect(
                lambda checked=False, page_key=key: self.set_active_page(page_key)
            )
            group.addButton(button)
            layout.addWidget(button)
            self.nav_buttons[key] = button

        layout.addStretch(1)

        user_card = QFrame()
        user_card.setObjectName("sidebarUserCard")
        user_card_layout = QVBoxLayout(user_card)
        user_card_layout.setContentsMargins(18, 18, 18, 18)
        user_card_layout.setSpacing(6)

        user_title = QLabel("Logged In")
        user_title.setObjectName("userCardTitle")
        user_card_layout.addWidget(user_title)

        sidebar_user_label = QLabel(self.user["username"])
        sidebar_user_label.setObjectName("userCardValue")
        user_card_layout.addWidget(sidebar_user_label)

        sidebar_role_label = QLabel(self.user["role"])
        sidebar_role_label.setObjectName("sidebarCaption")
        sidebar_role_label.setWordWrap(True)
        user_card_layout.addWidget(sidebar_role_label)
        layout.addWidget(user_card)
        return sidebar

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("pageHeader")
        apply_shadow(header, blur=26, y_offset=8)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        text_column = QVBoxLayout()
        text_column.setSpacing(4)
        self.page_kicker = QLabel("Workspace")
        self.page_kicker.setObjectName("pageKicker")
        self.page_title = QLabel("")
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel("")
        self.page_subtitle.setObjectName("pageSubtitle")
        self.page_subtitle.setWordWrap(True)
        text_column.addWidget(self.page_kicker)
        text_column.addWidget(self.page_title)
        text_column.addWidget(self.page_subtitle)
        layout.addLayout(text_column, 1)

        right = QHBoxLayout()
        right.setSpacing(10)

        self.header_user_badge = QLabel(f"{self.user['username']} | {self.user['role']}")
        self.header_user_badge.setObjectName("badge")
        right.addWidget(self.header_user_badge)

        logout_button = QPushButton("Log Out")
        logout_button.clicked.connect(self.switch_account)
        right.addWidget(logout_button)

        refresh_button = QPushButton("Refresh Views")
        refresh_button.clicked.connect(self.refresh_all_views)
        right.addWidget(refresh_button)

        self.header_export_button = QPushButton("Export Current Report")
        self.header_export_button.setProperty("variant", "primary")
        self.header_export_button.clicked.connect(self.export_report)
        self.header_export_button.setEnabled(False)
        right.addWidget(self.header_export_button)

        layout.addLayout(right)
        return header

    def _build_pages(self) -> None:
        self.page_stack.addWidget(create_scroll_page(self._build_dashboard_page()))
        self.page_stack.addWidget(create_scroll_page(self._build_import_page()))
        self.page_stack.addWidget(create_scroll_page(self._build_learner_page()))
        self.page_stack.addWidget(create_scroll_page(self._build_cohort_page()))
        self.page_stack.addWidget(create_scroll_page(self._build_report_page()))
        if "users" in self.page_order:
            self.page_stack.addWidget(create_scroll_page(self._build_user_accounts_page()))

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        hero = SectionCard(
            "Student Pathway guidance",
            "The workspace keeps assessment intake, learner interpretation, and report generation in one offline desktop flow.",
            eyebrow="Operational Overview",
            accent=True,
        )
        hero_row = QHBoxLayout()
        hero_row.setSpacing(22)

        hero_right = QVBoxLayout()
        hero_right.setSpacing(12)
        learner_button = QPushButton("Open Learner Workspace")
        learner_button.setProperty("variant", "ghost")
        learner_button.clicked.connect(lambda: self.set_active_page("learner"))
        hero_right.addWidget(learner_button)
        import_button = QPushButton("Review Data Intake")
        import_button.setProperty("variant", "ghost")
        import_button.clicked.connect(lambda: self.set_active_page("import"))
        hero_right.addWidget(import_button)
        cohort_button = QPushButton("Inspect Cohort Insights")
        cohort_button.setProperty("variant", "ghost")
        cohort_button.clicked.connect(lambda: self.set_active_page("cohort"))
        hero_right.addWidget(cohort_button)
        hero_right.addStretch(1)
        hero_row.addStretch(1)
        hero_row.addLayout(hero_right)
        hero.body.addLayout(hero_row)
        layout.addWidget(hero)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(16)
        stats_grid.setVerticalSpacing(16)
        self.dashboard_stats = {
            "learners": StatCard("Learner Records", "Active learners in the local database."),
            "assessments": StatCard("Assessment Rows", "Validated rows currently stored."),
            "reports": StatCard("Stored Reports", "PDF reports generated and logged locally."),
            "dominant": StatCard(
                "Dominant Pathway", "Largest pathway share in the current cohort."
            ),
        }
        positions = [
            ("learners", 0, 0),
            ("assessments", 0, 1),
            ("reports", 1, 0),
            ("dominant", 1, 1),
        ]
        for key, row, column in positions:
            stats_grid.addWidget(self.dashboard_stats[key], row, column)
        stats_grid.setColumnStretch(0, 1)
        stats_grid.setColumnStretch(1, 1)
        stats_panel = QWidget()
        stats_panel.setLayout(stats_grid)
        layout.addWidget(stats_panel)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(18)
        chart_card = SectionCard(
            "Pathway distribution",
            "A quick view of how the current learner cohort is distributed across the three pathway families.",
            eyebrow="Cohort Lens",
        )
        self.cohort_canvas = CohortCanvas()
        chart_card.body.addWidget(self.cohort_canvas)
        bottom_row.addWidget(chart_card, 2)

        narrative_card = SectionCard(
            "Operational support",
            "These notes are generated from the latest cohort snapshot to support planning conversations.",
            eyebrow="What To Do Next",
        )
        self.dashboard_pathway_breakdown = QVBoxLayout()
        self.dashboard_pathway_breakdown.setSpacing(10)
        narrative_card.body.addLayout(self.dashboard_pathway_breakdown)
        self.dashboard_narrative = QTextBrowser()
        self.dashboard_narrative.setMinimumHeight(220)
        narrative_card.body.addWidget(self.dashboard_narrative)
        bottom_row.addWidget(narrative_card, 1)
        layout.addLayout(bottom_row)
        layout.addStretch(1)
        return page

    def _build_import_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        top_row = QHBoxLayout()
        top_row.setSpacing(18)

        import_card = SectionCard(
            "Validate before import",
            "Keep the intake flow reliable: preview the file first, then commit only clean learner assessment data.",
            eyebrow="Data Intake",
        )
        self.import_path_input = QLineEdit()
        self.import_path_input.setPlaceholderText("Choose a CSV file with learner assessments")
        path_row = QHBoxLayout()
        browse_button = QPushButton("Browse CSV")
        browse_button.setProperty("variant", "primary")
        browse_button.clicked.connect(self.select_import_file)
        preview_button = QPushButton("Validate File")
        preview_button.clicked.connect(self.preview_import_file)
        import_button = QPushButton("Import Assessed Records")
        import_button.clicked.connect(self.perform_import)
        path_row.addWidget(self.import_path_input, 1)
        path_row.addWidget(browse_button)
        path_row.addWidget(preview_button)
        path_row.addWidget(import_button)
        import_card.body.addLayout(path_row)

        self.import_status = QLabel()
        self.import_status.setObjectName("statusBanner")
        self.import_status.setWordWrap(True)
        set_banner_state(
            self.import_status,
            "Select a file to validate the structure, check the rows, and preview the data before import.",
            "info",
        )
        import_card.body.addWidget(self.import_status)

        import_stats = QGridLayout()
        import_stats.setHorizontalSpacing(14)
        import_stats.setVerticalSpacing(14)
        self.import_stats = {
            "rows": StatCard("Rows Ready", "Rows available in the validated preview."),
            "learners": StatCard("Learners In File", "Unique learners represented in the preview."),
            "window": StatCard("Assessment Window", "Date range found in the preview file."),
        }
        for idx, key in enumerate(("rows", "learners", "window")):
            import_stats.addWidget(self.import_stats[key], 0, idx)
        import_card.body.addLayout(import_stats)
        top_row.addWidget(import_card, 2)

        guide_card = SectionCard(
            "Import checklist",
            "The system is strict by design so that users get dependable results and clear failure feedback.",
            eyebrow="Operator Guidance",
        )
        guide_browser = QTextBrowser()
        guide_browser.setHtml(
            """
            <h3 style="margin-top:0;">Expected columns</h3>
            <p><code>learner_id</code>, <code>learner_name</code>, <code>grade</code>, <code>term</code>,
            <code>subject</code>, <code>score</code>, <code>assessment_date</code></p>
            <h3>Accepted subject handling</h3>
            <p><b>Kiswahili</b> and <b>Kenyan Sign Language</b> are normalized into one analysis slot so
            the learner analysis stays consistent.</p>
            <h3>HCI note</h3>
            <p>Validate first, import second. This reduces surprise, prevents silent failures, and gives
            staff an immediate confirmation of what the system is about to store.</p>
            """
        )
        guide_card.body.addWidget(guide_browser)
        top_row.addWidget(guide_card, 1)
        layout.addLayout(top_row)

        preview_card = SectionCard(
            "Preview table",
            "A first look at the validated rows before they are committed to the database.",
            eyebrow="Incoming Records",
        )
        self.import_preview_table = QTableWidget(0, 7)
        self.import_preview_table.setHorizontalHeaderLabels(
            ["Learner ID", "Learner Name", "Grade", "Term", "Subject", "Score", "Assessment Date"]
        )
        self.import_preview_table.setAlternatingRowColors(True)
        self.import_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.import_preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.import_preview_table.verticalHeader().setVisible(False)
        self.import_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        preview_card.body.addWidget(self.import_preview_table)
        layout.addWidget(preview_card)
        layout.addStretch(1)
        return page

    def _build_learner_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        selector_card = SectionCard(
            "Select a learner and inspect the recommendation",
            "Search by learner ID or name, then review the pathway recommendation and the evidence behind it.",
            eyebrow="Learner Selection",
        )
        selector_row = QHBoxLayout()
        selector_row.setSpacing(12)
        self.learner_search_input = QLineEdit()
        self.learner_search_input.setPlaceholderText("Filter learner by ID or name")
        self.learner_search_input.textChanged.connect(self.populate_learner_selector)
        self.learner_combo = QComboBox()
        self.learner_combo.currentIndexChanged.connect(lambda _index: self.load_selected_learner())
        refresh_button = QPushButton("Refresh Learner View")
        refresh_button.clicked.connect(self.load_selected_learner)
        selector_row.addWidget(self.learner_search_input, 1)
        selector_row.addWidget(self.learner_combo, 2)
        selector_row.addWidget(refresh_button)
        selector_card.body.addLayout(selector_row)
        layout.addWidget(selector_card)

        hero = SectionCard(
            "Current learner snapshot",
            "The selected learner's recommendation, readiness band, subject evidence, and guidance notes appear here.",
            eyebrow="Readiness Snapshot",
            accent=True,
        )
        hero_grid = QGridLayout()
        hero_grid.setHorizontalSpacing(18)
        hero_grid.setVerticalSpacing(14)
        self.learner_name_value = self._make_hero_metric("Learner", "-")
        self.learner_pathway_value = self._make_hero_metric("Recommended pathway", "-")
        self.learner_track_value = self._make_hero_metric("Top track", "-")
        self.learner_score_value = self._make_hero_metric("Readiness score", "-")
        self.learner_confidence_value = self._make_hero_metric("Readiness band", "-")
        self.learner_assessment_value = self._make_hero_metric("Assessments used", "-")
        hero_grid.addWidget(self.learner_name_value, 0, 0)
        hero_grid.addWidget(self.learner_pathway_value, 0, 1)
        hero_grid.addWidget(self.learner_track_value, 0, 2)
        hero_grid.addWidget(self.learner_score_value, 1, 0)
        hero_grid.addWidget(self.learner_confidence_value, 1, 1)
        hero_grid.addWidget(self.learner_assessment_value, 1, 2)
        hero.body.addLayout(hero_grid)
        layout.addWidget(hero)

        upper_row = QHBoxLayout()
        upper_row.setSpacing(18)

        probability_card = SectionCard(
            "Readiness view",
            "See pathway readiness and the leading track options for the selected learner.",
            eyebrow="Decision Evidence",
        )
        pathway_group = QGroupBox("Pathway readiness")
        pathway_layout = QVBoxLayout(pathway_group)
        pathway_layout.setContentsMargins(16, 18, 16, 16)
        self.pathway_probabilities_layout = QVBoxLayout()
        self.pathway_probabilities_layout.setSpacing(10)
        pathway_layout.addLayout(self.pathway_probabilities_layout)

        track_group = QGroupBox("Top track options")
        track_layout = QVBoxLayout(track_group)
        track_layout.setContentsMargins(16, 18, 16, 16)
        self.track_probabilities_layout = QVBoxLayout()
        self.track_probabilities_layout.setSpacing(10)
        track_layout.addLayout(self.track_probabilities_layout)

        probability_card.body.addWidget(pathway_group)
        probability_card.body.addWidget(track_group)
        upper_row.addWidget(probability_card, 2)

        explanation_column = QVBoxLayout()
        explanation_column.setSpacing(18)
        strength_card = SectionCard(
            "What is helping this recommendation",
            "These subject patterns currently support the recommendation.",
            eyebrow="Positive Signals",
        )
        self.strengths_browser = QTextBrowser()
        strength_card.body.addWidget(self.strengths_browser)
        explanation_column.addWidget(strength_card)

        limiting_card = SectionCard(
            "What needs support",
            "These areas may need attention before the learner is strongly ready for the track.",
            eyebrow="Watch Areas",
        )
        self.limiting_browser = QTextBrowser()
        limiting_card.body.addWidget(self.limiting_browser)
        explanation_column.addWidget(limiting_card)
        upper_row.addLayout(explanation_column, 1)
        layout.addLayout(upper_row)

        lower_row = QHBoxLayout()
        lower_row.setSpacing(18)
        subject_card = SectionCard(
            "Subject readiness table",
            "Use the recent mean, long-run mean, trend, and consistency to explain the learner profile clearly.",
            eyebrow="Subject Evidence",
        )
        self.subject_table = QTableWidget(0, 5)
        self.subject_table.setHorizontalHeaderLabels(
            ["Subject", "Mean", "Recent Mean", "Trend", "Consistency"]
        )
        self.subject_table.setAlternatingRowColors(True)
        self.subject_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subject_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subject_table.verticalHeader().setVisible(False)
        self.subject_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        subject_card.body.addWidget(self.subject_table)
        lower_row.addWidget(subject_card, 2)

        guidance_card = SectionCard(
            "Guidance notes",
            "Narrative suggestions that can be shared during counselling or parent-facing report preparation.",
            eyebrow="Recommended Action",
        )
        self.guidance_browser = QTextBrowser()
        guidance_card.body.addWidget(self.guidance_browser)
        lower_row.addWidget(guidance_card, 1)
        layout.addLayout(lower_row)
        layout.addStretch(1)
        return page

    def _build_cohort_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(16)
        self.cohort_stats = {
            "cohort_size": StatCard(
                "Cohort Size", "Learners currently available for cohort analytics."
            ),
            "dominant": StatCard("Leading Pathway", "Largest share of the current cohort."),
            "balance": StatCard(
                "Pathway Spread", "Whether the cohort is concentrated or evenly distributed."
            ),
        }
        for idx, key in enumerate(("cohort_size", "dominant", "balance")):
            stats_grid.addWidget(self.cohort_stats[key], 0, idx)
        stats_host = QWidget()
        stats_host.setLayout(stats_grid)
        layout.addWidget(stats_host)

        row = QHBoxLayout()
        row.setSpacing(18)
        chart_card = SectionCard(
            "Distribution chart",
            "This view helps school staff identify the broad direction of the cohort before drilling into individuals.",
            eyebrow="Visual Summary",
        )
        self.cohort_page_canvas = CohortCanvas()
        chart_card.body.addWidget(self.cohort_page_canvas)
        row.addWidget(chart_card, 2)

        breakdown_card = SectionCard(
            "Breakdown and interpretation",
            "Each pathway summary below can guide where teachers may want to focus additional support or conversation.",
            eyebrow="Interpretation Layer",
        )
        self.cohort_breakdown_layout = QVBoxLayout()
        self.cohort_breakdown_layout.setSpacing(10)
        breakdown_card.body.addLayout(self.cohort_breakdown_layout)
        self.cohort_narrative = QTextBrowser()
        breakdown_card.body.addWidget(self.cohort_narrative)
        row.addWidget(breakdown_card, 1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        banner = SectionCard(
            "Readable reports for guidance conversations",
            "Use the current learner recommendation to generate a concise parent-facing PDF while keeping the full technical evidence in the app.",
            eyebrow="Report Export",
            accent=True,
        )
        banner_row = QHBoxLayout()
        banner_row.setSpacing(20)
        self.report_current_learner = self._make_hero_metric("Current learner", "-")
        self.report_current_pathway = self._make_hero_metric("Current pathway", "-")
        self.report_current_track = self._make_hero_metric("Current track", "-")
        banner_row.addWidget(self.report_current_learner)
        banner_row.addWidget(self.report_current_pathway)
        banner_row.addWidget(self.report_current_track)
        banner.body.addLayout(banner_row)

        banner_actions = QHBoxLayout()
        banner_actions.addStretch(1)
        self.page_export_button = QPushButton("Export Current Learner Report")
        self.page_export_button.setProperty("variant", "ghost")
        self.page_export_button.clicked.connect(self.export_report)
        self.page_export_button.setEnabled(False)
        banner_actions.addWidget(self.page_export_button)
        banner.body.addLayout(banner_actions)
        layout.addWidget(banner)

        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        preview_card = SectionCard(
            "Report preview",
            "Review the narrative that will shape the exported learner report.",
            eyebrow="Preview",
        )
        self.report_preview_browser = QTextBrowser()
        preview_card.body.addWidget(self.report_preview_browser)
        content_row.addWidget(preview_card, 2)

        meta_card = SectionCard(
            "Export details",
            "Confirm the current learner, report state, and saved output location before sharing the document.",
            eyebrow="Controls",
        )
        self.report_status_banner = QLabel()
        self.report_status_banner.setObjectName("statusBanner")
        self.report_status_banner.setWordWrap(True)
        set_banner_state(
            self.report_status_banner,
            "Select a learner in the workspace to prepare the report center.",
            "info",
        )
        meta_card.body.addWidget(self.report_status_banner)
        self.report_meta_browser = QTextBrowser()
        meta_card.body.addWidget(self.report_meta_browser)
        content_row.addWidget(meta_card, 1)
        layout.addLayout(content_row)
        layout.addStretch(1)
        return page

    def _build_user_accounts_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 24)
        layout.setSpacing(18)

        overview_row = QHBoxLayout()
        overview_row.setSpacing(18)

        overview = SectionCard(
            "Teacher access management",
            "Create additional teacher accounts so different staff members can sign in with their own credentials.",
            eyebrow="Admin Controls",
        )
        overview_row.addWidget(overview, 1)

        backup_card = SectionCard(
            "Database backup",
            "Save a point-in-time copy of all learner records and reports.",
            eyebrow="Maintenance",
        )
        self.backup_db_button = QPushButton("Backup database now")
        self.backup_db_button.setProperty("variant", "primary")
        self.backup_db_button.clicked.connect(self.run_database_backup)
        backup_card.body.addWidget(self.backup_db_button)
        overview_row.addWidget(backup_card, 0)

        layout.addLayout(overview_row)

        top_row = QHBoxLayout()
        top_row.setSpacing(18)

        create_card = SectionCard(
            "Create teacher account",
            "Accounts created here use the Teacher/Counsellor role and can access the normal workspace features.",
            eyebrow="New Account",
        )
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        self.account_username_input = QLineEdit()
        self.account_username_input.setPlaceholderText("New teacher username")
        self.account_password_input = QLineEdit()
        self.account_password_input.setPlaceholderText("Temporary password")
        self.account_password_input.setEchoMode(QLineEdit.Password)
        self.account_confirm_input = QLineEdit()
        self.account_confirm_input.setPlaceholderText("Confirm temporary password")
        self.account_confirm_input.setEchoMode(QLineEdit.Password)
        form.addRow("Username", self.account_username_input)
        form.addRow("Password", self.account_password_input)
        form.addRow("Confirm password", self.account_confirm_input)
        create_card.body.addLayout(form)

        self.account_status_banner = QLabel()
        self.account_status_banner.setObjectName("statusBanner")
        self.account_status_banner.setWordWrap(True)
        set_banner_state(
            self.account_status_banner,
            "Create teacher accounts here. Staff can then use Log Out to switch and sign in with the new credentials.",
            "info",
        )
        create_card.body.addWidget(self.account_status_banner)

        create_button = QPushButton("Create Teacher Account")
        create_button.setProperty("variant", "primary")
        create_button.clicked.connect(self.create_teacher_account)
        create_card.body.addWidget(create_button)
        top_row.addWidget(create_card, 1)

        manage_card = SectionCard(
            "Manage selected account",
            "Reset a teacher password, deactivate access, reactivate an account, or remove an old account.",
            eyebrow="Account Actions",
        )
        self.selected_account_label = QLabel("No teacher account selected yet.")
        self.selected_account_label.setObjectName("sectionDescription")
        self.selected_account_label.setWordWrap(True)
        manage_card.body.addWidget(self.selected_account_label)

        manage_form = QFormLayout()
        manage_form.setHorizontalSpacing(16)
        manage_form.setVerticalSpacing(12)
        self.manage_password_input = QLineEdit()
        self.manage_password_input.setPlaceholderText("New temporary password")
        self.manage_password_input.setEchoMode(QLineEdit.Password)
        self.manage_confirm_input = QLineEdit()
        self.manage_confirm_input.setPlaceholderText("Confirm new temporary password")
        self.manage_confirm_input.setEchoMode(QLineEdit.Password)
        manage_form.addRow("New password", self.manage_password_input)
        manage_form.addRow("Confirm password", self.manage_confirm_input)
        manage_card.body.addLayout(manage_form)

        self.account_action_banner = QLabel()
        self.account_action_banner.setObjectName("statusBanner")
        self.account_action_banner.setWordWrap(True)
        set_banner_state(
            self.account_action_banner,
            "Select a teacher account in the table to manage its access.",
            "info",
        )
        manage_card.body.addWidget(self.account_action_banner)

        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(12)
        action_grid.setVerticalSpacing(12)
        self.reset_password_button = QPushButton("Reset Password")
        self.reset_password_button.clicked.connect(self.reset_selected_teacher_password)
        self.deactivate_account_button = QPushButton("Deactivate Account")
        self.deactivate_account_button.clicked.connect(
            lambda: self.change_selected_teacher_active_state(False)
        )
        self.reactivate_account_button = QPushButton("Reactivate Account")
        self.reactivate_account_button.clicked.connect(
            lambda: self.change_selected_teacher_active_state(True)
        )
        self.delete_account_button = QPushButton("Delete Account")
        self.delete_account_button.clicked.connect(self.delete_selected_teacher_account)
        action_grid.addWidget(self.reset_password_button, 0, 0)
        action_grid.addWidget(self.deactivate_account_button, 0, 1)
        action_grid.addWidget(self.reactivate_account_button, 1, 0)
        action_grid.addWidget(self.delete_account_button, 1, 1)
        manage_card.body.addLayout(action_grid)
        top_row.addWidget(manage_card, 1)

        layout.addLayout(top_row)

        table_card = SectionCard(
            "Current accounts",
            "Review all local user accounts currently stored in the desktop application database.",
            eyebrow="Access Overview",
        )
        self.user_accounts_table = QTableWidget(0, 3)
        self.user_accounts_table.setHorizontalHeaderLabels(["Username", "Role", "Status"])
        self.user_accounts_table.setAlternatingRowColors(True)
        self.user_accounts_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.user_accounts_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_accounts_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.user_accounts_table.verticalHeader().setVisible(False)
        self.user_accounts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_accounts_table.itemSelectionChanged.connect(self.refresh_account_action_state)
        table_card.body.addWidget(self.user_accounts_table)
        layout.addWidget(table_card)
        layout.addStretch(1)
        return page

    def _make_hero_metric(self, label: str, value: str) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("reportBanner")
        wrapper.setStyleSheet(
            """
            QFrame#reportBanner {
                background: rgba(6, 20, 34, 0.22);
                border: 1px solid rgba(255, 250, 243, 0.32);
                border-radius: 18px;
            }
            """
        )
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        title = QLabel(label)
        title.setObjectName("heroMetricLabel")
        content = QLabel(value)
        content.setObjectName("heroMetricValue")
        content.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(content)
        wrapper.value_label = content  # type: ignore[attr-defined]
        return wrapper

    def _set_hero_metric(self, widget: QWidget, value: str) -> None:
        widget.value_label.setText(value)  # type: ignore[attr-defined]

    def set_active_page(self, page_key: str) -> None:
        self.page_stack.setCurrentIndex(self.page_order.index(page_key))
        for key, button in self.nav_buttons.items():
            button.setChecked(key == page_key)
        title, subtitle = PAGE_METADATA[page_key]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)

    def refresh_all_views(self) -> None:
        previous_id = self.current_learner_id or self.selected_learner_id()
        self.all_assessments = load_all_assessments()
        self.learner_records = load_learners()
        self.latest_saved_report = latest_report_path()
        if "users" in self.page_order:
            self.refresh_user_accounts()
        self.refresh_dashboard()
        self.refresh_import_summary()
        self.populate_learner_selector(previous_id)
        self.refresh_cohort_chart()

        selected_id = previous_id or self.selected_learner_id()
        if selected_id:
            self.load_selected_learner(selected_id)
        elif self.learner_records:
            self.load_selected_learner(self.learner_records[0]["learner_id"])
        else:
            self.clear_learner_state()
            self.refresh_report_center()

        self.statusBar().showMessage("Views refreshed from local storage.")

    def selected_user_record(self) -> dict[str, object] | None:
        if "users" not in self.page_order or not hasattr(self, "user_accounts_table"):
            return None
        row_index = self.user_accounts_table.currentRow()
        if row_index < 0 or row_index >= len(self.user_records):
            return None
        return self.user_records[row_index]

    def refresh_account_action_state(self) -> None:
        if "users" not in self.page_order or not hasattr(self, "selected_account_label"):
            return
        record = self.selected_user_record()
        is_teacher = bool(record) and record["role"] == "Teacher/Counsellor"
        is_active = bool(record and record["is_active"])

        self.reset_password_button.setEnabled(is_teacher)
        self.deactivate_account_button.setEnabled(is_teacher and is_active)
        self.reactivate_account_button.setEnabled(is_teacher and not is_active)
        self.delete_account_button.setEnabled(is_teacher)

        if not record:
            self.selected_account_label.setText("No teacher account selected yet.")
            set_banner_state(
                self.account_action_banner,
                "Select a teacher account in the table to manage its access.",
                "info",
            )
            return

        self.selected_account_label.setText(
            f"Selected account: {record['username']} | {record['role']} | {record['status']}"
        )
        if not is_teacher:
            set_banner_state(
                self.account_action_banner,
                "Admin accounts are protected here. Select a teacher account to use these actions.",
                "info",
            )

    def refresh_user_accounts(self, preserve_username: str | None = None) -> None:
        if "users" not in self.page_order or not hasattr(self, "user_accounts_table"):
            return
        preserve_username = preserve_username or (
            self.selected_user_record()["username"] if self.selected_user_record() else None
        )
        self.user_records = list_users()
        table = self.user_accounts_table
        table.setRowCount(len(self.user_records))
        for row_index, record in enumerate(self.user_records):
            username_item = QTableWidgetItem(record["username"])
            role_item = QTableWidgetItem(record["role"])
            status_item = QTableWidgetItem(record["status"])
            table.setItem(row_index, 0, username_item)
            table.setItem(row_index, 1, role_item)
            table.setItem(row_index, 2, status_item)
            status_item.setTextAlignment(Qt.AlignCenter)
        table.resizeRowsToContents()
        if preserve_username:
            for row_index, record in enumerate(self.user_records):
                if record["username"] == preserve_username:
                    table.selectRow(row_index)
                    break
        self.refresh_account_action_state()

    def create_teacher_account(self) -> None:
        if self.user.get("role") != "Admin":
            QMessageBox.warning(
                self, "Access denied", "Only admin accounts can create additional teacher users."
            )
            return

        username = self.account_username_input.text().strip()
        password = self.account_password_input.text()
        confirm = self.account_confirm_input.text()

        if password != confirm:
            set_banner_state(
                self.account_status_banner,
                "The password confirmation does not match. Re-enter the same password in both fields.",
                "error",
            )
            return

        try:
            created_user = create_user(username, password, "Teacher/Counsellor")
        except ValueError as exc:
            set_banner_state(self.account_status_banner, str(exc), "error")
            return

        self.account_username_input.clear()
        self.account_password_input.clear()
        self.account_confirm_input.clear()
        set_banner_state(
            self.account_status_banner,
            f"Teacher account '{created_user['username']}' was created successfully. The user can now sign in from the login screen.",
            "success",
        )
        self.refresh_user_accounts(created_user["username"])
        self.statusBar().showMessage(f"User account {created_user['username']} created.")

    def reset_selected_teacher_password(self) -> None:
        record = self.selected_user_record()
        if not record or record["role"] != "Teacher/Counsellor":
            QMessageBox.information(
                self,
                "Select teacher account",
                "Select a teacher account before resetting a password.",
            )
            return

        password = self.manage_password_input.text()
        confirm = self.manage_confirm_input.text()
        if password != confirm:
            set_banner_state(
                self.account_action_banner,
                "The new password and confirmation do not match.",
                "error",
            )
            return

        try:
            reset_user_password(str(record["username"]), password)
        except ValueError as exc:
            set_banner_state(self.account_action_banner, str(exc), "error")
            return

        self.manage_password_input.clear()
        self.manage_confirm_input.clear()
        set_banner_state(
            self.account_action_banner,
            f"Password reset completed for {record['username']}. Share the new temporary password securely with the teacher.",
            "success",
        )
        self.statusBar().showMessage(f"Password reset for {record['username']}.")

    def change_selected_teacher_active_state(self, is_active: bool) -> None:
        record = self.selected_user_record()
        if not record or record["role"] != "Teacher/Counsellor":
            QMessageBox.information(
                self,
                "Select teacher account",
                "Select a teacher account before changing access status.",
            )
            return

        current_active = bool(record["is_active"])
        if current_active == is_active:
            set_banner_state(
                self.account_action_banner,
                f"{record['username']} is already marked as {'active' if is_active else 'inactive'}.",
                "info",
            )
            return

        action_label = "reactivate" if is_active else "deactivate"
        if (
            QMessageBox.question(
                self,
                "Confirm account change",
                f"Do you want to {action_label} the account '{record['username']}'?",
            )
            != QMessageBox.Yes
        ):
            return

        try:
            set_user_active(str(record["username"]), is_active)
        except ValueError as exc:
            set_banner_state(self.account_action_banner, str(exc), "error")
            return

        self.refresh_user_accounts(str(record["username"]))
        set_banner_state(
            self.account_action_banner,
            f"Account '{record['username']}' was {'reactivated' if is_active else 'deactivated'} successfully.",
            "success",
        )
        self.statusBar().showMessage(f"Updated access for {record['username']}.")

    def delete_selected_teacher_account(self) -> None:
        record = self.selected_user_record()
        if not record or record["role"] != "Teacher/Counsellor":
            QMessageBox.information(
                self, "Select teacher account", "Select a teacher account before deleting it."
            )
            return

        if (
            QMessageBox.question(
                self,
                "Delete teacher account",
                f"Delete the account '{record['username']}' permanently?\n\nThis cannot be undone.",
            )
            != QMessageBox.Yes
        ):
            return

        try:
            delete_user(str(record["username"]))
        except ValueError as exc:
            set_banner_state(self.account_action_banner, str(exc), "error")
            return

        self.refresh_user_accounts()
        set_banner_state(
            self.account_action_banner,
            f"Account '{record['username']}' was deleted successfully.",
            "success",
        )
        self.statusBar().showMessage(f"Updated access for {record['username']}.")

    def run_database_backup(self) -> None:
        try:
            path = backup_database()
            QMessageBox.information(
                self,
                "Backup Successful",
                f"The database was successfully backed up to:\n\n{path}",
            )
            log.info("Database backed up by user request.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Backup Failed", f"Could not back up database: {exc}")
            log.error("Database backup failed: %s", exc)

    def refresh_dashboard(self) -> None:
        assessment_count = len(self.all_assessments)
        learner_count = (
            self.all_assessments["learner_id"].nunique() if not self.all_assessments.empty else 0
        )
        self.dashboard_stats["learners"].set_content(
            str(learner_count),
            "Learners currently available for review and support.",
        )
        self.dashboard_stats["assessments"].set_content(
            f"{assessment_count:,}",
            "Assessment rows stored after validation and normalization.",
        )
        self.dashboard_stats["reports"].set_content(
            str(count_reports()),
            "Reports saved in the local report registry.",
        )
        if self.cohort_counts:
            dominant = max(self.cohort_counts, key=self.cohort_counts.get)
            share = (self.cohort_counts[dominant] / sum(self.cohort_counts.values())) * 100
            self.dashboard_stats["dominant"].set_content(
                dominant, f"{share:.1f}% of the current cohort."
            )
        else:
            self.dashboard_stats["dominant"].set_content(
                "No data", "Import learner records to generate a cohort view."
            )

    def refresh_import_summary(self) -> None:
        self.import_stats["rows"].set_content("-", "Preview a file to inspect the validated rows.")
        self.import_stats["learners"].set_content(
            "-", "Preview a file to count represented learners."
        )
        self.import_stats["window"].set_content(
            "-", "Preview a file to inspect the assessment window."
        )

    def populate_learner_selector(self, preserve_id: str | None = None) -> None:
        preserve_id = preserve_id or self.current_learner_id
        query = (
            self.learner_search_input.text().strip().lower()
            if hasattr(self, "learner_search_input")
            else ""
        )
        filtered = [
            learner
            for learner in self.learner_records
            if not query
            or query in learner["learner_id"].lower()
            or query in learner["learner_name"].lower()
        ]

        self.learner_combo.blockSignals(True)
        self.learner_combo.clear()
        for learner in filtered:
            self.learner_combo.addItem(
                f"{learner['learner_id']}  |  {learner['learner_name']}",
                learner["learner_id"],
            )
        if filtered:
            target = (
                preserve_id
                if any(item["learner_id"] == preserve_id for item in filtered)
                else filtered[0]["learner_id"]
            )
            index = self.learner_combo.findData(target)
            if index >= 0:
                self.learner_combo.setCurrentIndex(index)
        self.learner_combo.blockSignals(False)

    def selected_learner_id(self) -> str | None:
        current = self.learner_combo.currentData()
        return str(current) if current else None

    def select_import_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Assessment CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self.import_path_input.setText(path)
            self.preview_import_file()

    def preview_import_file(self) -> None:
        path = self.import_path_input.text().strip()
        if not path:
            set_banner_state(self.import_status, "Choose a CSV file before validation.", "error")
            return
        try:
            frame = load_assessment_csv(path)
        except IngestionError as exc:
            self.import_preview_frame = None
            self.import_preview_source = None
            self.import_preview_table.setRowCount(0)
            self.refresh_import_summary()
            # Show all row-level errors so the user can fix the whole file at once.
            if exc.row_errors:
                error_detail = "\n".join(exc.row_errors[:20])  # cap at 20 for display
                message = f"{exc}\n\nProblems found:\n{error_detail}"
                if len(exc.row_errors) > 20:
                    message += f"\n… and {len(exc.row_errors) - 20} more."
            else:
                message = str(exc)
            set_banner_state(self.import_status, message, "error")
            log.warning("Import validation failed for '%s': %s", path, exc)
            self.statusBar().showMessage("Validation failed.")
            return
        except Exception as exc:
            self.import_preview_frame = None
            self.import_preview_source = None
            self.import_preview_table.setRowCount(0)
            self.refresh_import_summary()
            set_banner_state(self.import_status, f"Could not read file: {exc}", "error")
            log.error("Unexpected error reading '%s': %s", path, exc, exc_info=True)
            self.statusBar().showMessage("Validation failed.")
            return

        self.import_preview_frame = frame
        self.import_preview_source = path
        self.populate_table(
            self.import_preview_table,
            frame.head(10),
            ["learner_id", "learner_name", "grade", "term", "subject", "score", "assessment_date"],
        )
        self.import_stats["rows"].set_content(
            f"{len(frame):,}", "Validated rows now ready for import."
        )
        self.import_stats["learners"].set_content(
            str(frame["learner_id"].nunique()),
            "Unique learners represented in this file.",
        )
        self.import_stats["window"].set_content(
            f"{frame['assessment_date'].min()} to {frame['assessment_date'].max()}",
            "Assessment dates in the validated file.",
        )
        set_banner_state(
            self.import_status,
            f"Validation complete. {len(frame):,} rows are ready from {Path(path).name}.",
            "success",
        )
        self.statusBar().showMessage("Import file validated successfully.")

    def perform_import(self) -> None:
        path = self.import_path_input.text().strip()
        if not path:
            set_banner_state(self.import_status, "Choose a CSV file before importing.", "error")
            return
        if self.import_preview_frame is None or self.import_preview_source != path:
            self.preview_import_file()
        if self.import_preview_frame is None:
            return
        try:
            imported = import_assessment_frame(self.import_preview_frame)
        except IngestionError as exc:
            set_banner_state(self.import_status, f"Import rejected: {exc}", "error")
            log.error("Import failed: %s", exc)
            self.statusBar().showMessage("Import failed.")
            return
        except Exception as exc:
            set_banner_state(self.import_status, f"Database write failed: {exc}", "error")
            log.error("Unexpected import error: %s", exc, exc_info=True)
            self.statusBar().showMessage("Import failed.")
            return

        set_banner_state(
            self.import_status,
            f"Import complete. {imported:,} rows from {Path(path).name} are now stored in the local database.",
            "success",
        )
        self.statusBar().showMessage("Assessment rows imported successfully.")
        self.refresh_all_views()

    def load_selected_learner(self, learner_id: str | None = None) -> None:
        learner_id = learner_id or self.selected_learner_id()
        if not learner_id:
            self.clear_learner_state()
            self.refresh_report_center()
            return

        target_index = self.learner_combo.findData(learner_id)
        if target_index >= 0 and self.learner_combo.currentIndex() != target_index:
            self.learner_combo.blockSignals(True)
            self.learner_combo.setCurrentIndex(target_index)
            self.learner_combo.blockSignals(False)

        assessments = load_assessments_for_learner(learner_id)
        if assessments.empty:
            self.clear_learner_state()
            self.refresh_report_center()
            return

        subject_summary = build_subject_summary(assessments)
        prediction = predict_for_learner(assessments, self.bundle)
        save_prediction(
            learner_id=prediction.learner_id,
            predicted_pathway=prediction.predicted_pathway,
            predicted_track=prediction.predicted_track,
            top_track_score=prediction.top_track_score,
            pathway_probabilities=prediction.pathway_probabilities,
            track_probabilities=prediction.track_probabilities,
            guidance_notes=prediction.guidance_notes,
            strengths=prediction.strengths,
            limiting_factors=prediction.limiting_factors,
            feature_importance=prediction.feature_importance,
        )
        self.current_prediction = prediction
        self.current_learner_id = learner_id
        self.current_subject_summary = subject_summary
        self.render_learner_prediction(prediction, subject_summary, len(assessments))
        self.refresh_report_center()
        self.header_export_button.setEnabled(True)
        self.page_export_button.setEnabled(True)
        self.statusBar().showMessage(f"Learner {prediction.learner_id} loaded.")

    def clear_learner_state(self) -> None:
        self.current_prediction = None
        self.current_learner_id = None
        self.current_subject_summary = {}
        for widget in (
            self.learner_name_value,
            self.learner_pathway_value,
            self.learner_track_value,
            self.learner_score_value,
            self.learner_confidence_value,
            self.learner_assessment_value,
        ):
            self._set_hero_metric(widget, "-")
        clear_layout(self.pathway_probabilities_layout)
        clear_layout(self.track_probabilities_layout)
        self.strengths_browser.setHtml("<p>No learner selected yet.</p>")
        self.limiting_browser.setHtml("<p>No learner selected yet.</p>")
        self.guidance_browser.setHtml("<p>No learner selected yet.</p>")
        self.subject_table.setRowCount(0)
        self.header_export_button.setEnabled(False)
        self.page_export_button.setEnabled(False)

    def render_learner_prediction(
        self,
        prediction: PredictionOutput,
        subject_summary: dict[str, dict[str, float]],
        assessment_rows: int,
    ) -> None:
        dominant_probability = max(prediction.pathway_probabilities.values())
        score_text = (
            f"{prediction.top_track_score * 100:.1f}%"
            if 0.0 <= prediction.top_track_score <= 1.0
            else f"{prediction.top_track_score:.2f}"
        )
        self._set_hero_metric(
            self.learner_name_value, f"{prediction.learner_name} ({prediction.learner_id})"
        )
        self._set_hero_metric(self.learner_pathway_value, prediction.predicted_pathway)
        self._set_hero_metric(self.learner_track_value, prediction.predicted_track)
        self._set_hero_metric(self.learner_score_value, score_text)
        self._set_hero_metric(
            self.learner_confidence_value,
            f"{readiness_band(dominant_probability)} | {dominant_probability * 100:.1f}%",
        )
        self._set_hero_metric(self.learner_assessment_value, str(assessment_rows))

        self.populate_probability_group(
            self.pathway_probabilities_layout,
            prediction.pathway_probabilities,
            PATHWAY_COLORS,
            lambda label, value: f"{readiness_band(value)} pathway readiness",
        )
        top_tracks = dict(
            sorted(prediction.track_probabilities.items(), key=lambda item: item[1], reverse=True)[
                :4
            ]
        )
        self.populate_probability_group(
            self.track_probabilities_layout,
            top_tracks,
            TRACK_COLORS,
            lambda label, value: f"{readiness_band(value)} track readiness for {label}",
        )
        self.strengths_browser.setHtml(self._factor_html(prediction.strengths, positive=True))
        self.limiting_browser.setHtml(
            self._factor_html(prediction.limiting_factors, positive=False)
        )
        self.guidance_browser.setHtml(self._guidance_html(prediction))
        self.populate_subject_table(subject_summary, prediction)

    def populate_probability_group(
        self,
        layout: QVBoxLayout,
        probabilities: dict[str, float],
        color_map: dict[str, str],
        note_builder,
    ) -> None:
        clear_layout(layout)
        if not probabilities:
            placeholder = QLabel("No readiness data available yet.")
            placeholder.setStyleSheet("color: #42586a;")
            layout.addWidget(placeholder)
            return
        for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            bar = ProbabilityBar(color_map.get(label, "#2d6d68"))
            bar.set_content(label, value, note_builder(label, value))
            layout.addWidget(bar)
        layout.addStretch(1)

    def populate_subject_table(
        self, subject_summary: dict[str, dict[str, float]], prediction: PredictionOutput
    ) -> None:
        strengths = {item["subject"] for item in prediction.strengths}
        limits = {item["subject"] for item in prediction.limiting_factors}
        ordered_subjects = sorted(
            subject_summary.items(), key=lambda item: item[1]["recent_mean"], reverse=True
        )
        self.subject_table.setRowCount(len(ordered_subjects))
        for row, (subject, stats) in enumerate(ordered_subjects):
            trend_text, trend_color = trend_label(stats["trend"])
            values = [
                subject,
                f"{stats['mean']:.1f}",
                f"{stats['recent_mean']:.1f}",
                f"{trend_text} ({stats['trend']:+.2f})",
                f"{stats['consistency']:.1f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 3:
                    item.setForeground(QColor(trend_color))
                if subject in strengths:
                    item.setBackground(QColor("#e8f5ee"))
                elif subject in limits:
                    item.setBackground(QColor("#fdecea"))
                self.subject_table.setItem(row, column, item)
        self.subject_table.resizeRowsToContents()

    def _factor_html(self, factors: list[dict[str, float | str]], *, positive: bool) -> str:
        if not factors:
            return "<p>No clear subject signals were identified for this category yet.</p>"
        tone = (
            "is currently supporting this recommendation."
            if positive
            else "may need more support before readiness becomes stronger."
        )
        items = "".join(
            f"<li><b>{escape(str(item['subject']))}</b> {tone}</li>" for item in factors
        )
        return f"<ul>{items}</ul>"

    def _guidance_html(self, prediction: PredictionOutput) -> str:
        items = "".join(f"<li>{escape(note)}</li>" for note in prediction.guidance_notes)
        return (
            f"<p><b>Recommendation summary:</b> {escape(prediction.predicted_pathway)} with "
            f"{escape(prediction.predicted_track)} as the strongest track.</p><ul>{items}</ul>"
        )

    def refresh_cohort_chart(self) -> None:
        if self.all_assessments.empty:
            self.cohort_counts = Counter()
            self.cohort_canvas.render_counts(self.cohort_counts)
            self.cohort_page_canvas.render_counts(self.cohort_counts)
            self.cohort_stats["cohort_size"].set_content(
                "0", "Import learner assessment data to activate cohort analytics."
            )
            self.cohort_stats["dominant"].set_content(
                "No data", "No pathway prediction is available until records are present."
            )
            self.cohort_stats["balance"].set_content(
                "N/A", "No cohort spread can be calculated yet."
            )
            clear_layout(self.dashboard_pathway_breakdown)
            clear_layout(self.cohort_breakdown_layout)
            self.dashboard_narrative.setHtml("<p>No cohort data is available yet.</p>")
            self.cohort_narrative.setHtml("<p>No cohort data is available yet.</p>")
            return

        feature_frame = build_feature_matrix(self.all_assessments)
        predicted_tracks = self.bundle["model"].predict(
            feature_frame[self.bundle["feature_columns"]]
        )
        counts = Counter(pathway_from_track(track) for track in predicted_tracks)
        self.cohort_counts = counts
        total = sum(counts.values())

        self.cohort_canvas.render_counts(counts)
        self.cohort_page_canvas.render_counts(counts)
        dominant = max(counts, key=counts.get)
        dominant_share = (counts[dominant] / total) * 100
        spread = max(counts.values()) - min(counts.values())
        spread_label = "Balanced" if spread <= max(4, total * 0.08) else "Concentrated"

        self.dashboard_stats["dominant"].set_content(
            dominant, f"{dominant_share:.1f}% of the current cohort."
        )
        self.cohort_stats["cohort_size"].set_content(
            str(total), "Learners currently represented in cohort analytics."
        )
        self.cohort_stats["dominant"].set_content(
            dominant, f"{dominant_share:.1f}% of learners lean toward this pathway."
        )
        self.cohort_stats["balance"].set_content(
            spread_label, f"Difference between the highest and lowest pathway counts: {spread}."
        )
        self.populate_pathway_breakdown(self.dashboard_pathway_breakdown, counts)
        self.populate_pathway_breakdown(self.cohort_breakdown_layout, counts)
        narrative = self._cohort_narrative_html(counts)
        self.dashboard_narrative.setHtml(narrative)
        self.cohort_narrative.setHtml(narrative)

    def populate_pathway_breakdown(self, layout: QVBoxLayout, counts: Counter[str]) -> None:
        clear_layout(layout)
        total = sum(counts.values()) or 1
        for pathway in PATHWAY_COLORS:
            if pathway not in counts:
                continue
            share = counts[pathway] / total
            bar = ProbabilityBar(PATHWAY_COLORS[pathway])
            bar.set_content(
                pathway, share, f"{counts[pathway]} learners | {share * 100:.1f}% of the cohort"
            )
            layout.addWidget(bar)
        layout.addStretch(1)

    def _cohort_narrative_html(self, counts: Counter[str]) -> str:
        total = sum(counts.values()) or 1
        dominant = max(counts, key=counts.get)
        dominant_share = counts[dominant] / total
        return (
            "<ul>"
            f"<li><b>{escape(dominant)}</b> currently leads with {dominant_share * 100:.1f}% of the cohort.</li>"
            "<li>Use this view to plan broad intervention themes before moving into individual learner counselling.</li>"
            "</ul>"
        )

    def refresh_report_center(self) -> None:
        if not self.current_prediction:
            for widget in (
                self.report_current_learner,
                self.report_current_pathway,
                self.report_current_track,
            ):
                self._set_hero_metric(widget, "-")
            self.report_preview_browser.setHtml(
                "<p>Select a learner in the workspace to prepare a report.</p>"
            )
            self.report_meta_browser.setHtml("<p>No report-ready learner is currently loaded.</p>")
            set_banner_state(
                self.report_status_banner,
                "Select a learner in the workspace to prepare the report center.",
                "info",
            )
            return

        prediction = self.current_prediction
        self._set_hero_metric(
            self.report_current_learner, f"{prediction.learner_name} ({prediction.learner_id})"
        )
        self._set_hero_metric(self.report_current_pathway, prediction.predicted_pathway)
        self._set_hero_metric(self.report_current_track, prediction.predicted_track)
        pathway_items = "".join(
            f"<li>{escape(label)}: {value * 100:.1f}%</li>"
            for label, value in sorted(
                prediction.pathway_probabilities.items(), key=lambda item: item[1], reverse=True
            )
        )
        track_items = "".join(
            f"<li>{escape(label)}: {value * 100:.1f}%</li>"
            for label, value in sorted(
                prediction.track_probabilities.items(), key=lambda item: item[1], reverse=True
            )[:4]
        )
        notes = "".join(f"<li>{escape(note)}</li>" for note in prediction.guidance_notes)
        preview_score_text = (
            f"{prediction.top_track_score * 100:.1f}%"
            if 0.0 <= prediction.top_track_score <= 1.0
            else f"{prediction.top_track_score:.2f}"
        )
        self.report_preview_browser.setHtml(
            f"<h3 style='margin-top:0;'>Readiness summary for {escape(prediction.learner_name)}</h3>"
            f"<p><b>Recommended pathway:</b> {escape(prediction.predicted_pathway)}<br>"
            f"<b>Top track:</b> {escape(prediction.predicted_track)}<br>"
            f"<b>Readiness score:</b> {preview_score_text}</p>"
            f"<h4>Pathway readiness</h4><ul>{pathway_items}</ul>"
            f"<h4>Top track alternatives</h4><ul>{track_items}</ul>"
            f"<h4>Guidance notes</h4><ul>{notes}</ul>"
        )
        path_text = self.latest_saved_report or "No report has been exported yet."
        self.report_meta_browser.setHtml(
            f"<p><b>Current learner:</b> {escape(prediction.learner_id)}</p>"
            f"<p><b>Last saved report path:</b><br><code>{escape(path_text)}</code></p>"
            "<p><b>Guidance reminder:</b> The report is designed for explanation and discussion, not automatic placement.</p>"
        )
        set_banner_state(
            self.report_status_banner,
            f"The report center is ready for {prediction.learner_id}. Review the summary, then export the PDF.",
            "success",
        )

    def export_report(self) -> None:
        if not self.current_prediction:
            QMessageBox.information(self, "No learner selected", "Select a learner profile first.")
            return
        try:
            path = generate_pdf_report(self.current_prediction)
        except ReportError as exc:
            QMessageBox.warning(self, "Report Failed", f"Could not generate the report:\n{exc}")
            log.error("Report generation failed: %s", exc)
            return
        save_report_record(self.current_prediction.learner_id, path)
        self.latest_saved_report = str(path)
        self.refresh_report_center()
        self.refresh_dashboard()
        log.info("Report exported to %s", path)
        self.statusBar().showMessage(f"Report exported to {path}")
        QMessageBox.information(
            self,
            "Report Exported",
            f"The learner report was exported successfully.\n\nSaved to:\n{path}",
        )

    def populate_table(self, table: QTableWidget, frame: pd.DataFrame, columns: list[str]) -> None:
        """Fill a QTableWidget from a DataFrame, centering numeric columns."""
        table.setRowCount(len(frame))
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, column_name in enumerate(columns):
                item = QTableWidgetItem(str(row[column_name]))
                if column_name in {"score", "assessment_date"}:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_index, column_index, item)
        table.resizeRowsToContents()


def main() -> None:
    """Application entry point: seed data, then run login → workspace loop.

    The session loop allows users to log out and sign back in with a
    different account without restarting the process.  ``setQuitOnLastWindowClosed(False)``
    prevents Qt from exiting when the login dialog closes.
    """
    from .config import APP_LOG_PATH

    configure_logging(APP_LOG_PATH)
    ensure_seed_data()
    application = QApplication([])
    # Prevent Qt from quitting when the login dialog closes; we manage
    # the lifecycle explicitly via the session loop.
    application.setQuitOnLastWindowClosed(False)
    application.setStyle("Fusion")
    application.setFont(QFont("Candara", 10))
    application.setStyleSheet(APP_STYLESHEET)

    session_state: dict[str, MainWindow | None] = {"window": None}

    def finish_session(restart_session: bool) -> None:
        session_state["window"] = None
        if restart_session:
            QTimer.singleShot(0, start_session)
            return
        application.quit()

    def start_session() -> None:
        login = LoginDialog()
        if login.exec() != QDialog.Accepted or not login.user:
            application.quit()
            return
        window = MainWindow(login.user)
        session_state["window"] = window
        window.session_finished.connect(finish_session)
        window.show()

    start_session()
    application.exec()
