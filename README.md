# CBC Pathway Readiness eXplainable AI (cbc_xai)

**A desktop application prototype investigating transparent machine learning for the Kenyan Competency-Based Curriculum (CBC) pathway selection.**

> [!WARNING]
> **Research Prototype**
> This system was built for a final-year academic project. It is designed to operate on **synthetic, generated data** to demonstrate how Explainable AI (XAI) can support teacher-parent guidance conversations. It is **not** suitable for live deployment in schools to make automated placement decisions.

## 📖 Overview

As the Kenyan education system transitions to the Competency-Based Curriculum (CBC), learners face critical pathway and track choices at the end of Junior School (Grade 9). This project provides a lightweight, offline-first desktop application that acts as a decision-support tool for teachers, using:

1. **Domain Logic:** Rule-based composites following curriculum guidelines for tracks (e.g., STEM, Social Sciences, Arts and Sports Science).
2. **Machine Learning:** A Random Forest classifier trained on historical assessment patterns to predict track readiness.
3. **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) values to make model predictions transparent, identifying specific subject strengths and limiting factors for each learner.

## ✨ Features

- **Offline-First Desktop App:** Built with Python, PySide6, and SQLite. Designed to run on resource-constrained hardware without needing internet access.
- **Strict Data Ingestion:** Bulk import of learner assessments via CSV with strict, all-errors-at-once validation to help teachers fix data issues easily.
- **Automated ML Pipeline:** Includes a script to generate synthetic assessment data, perform stratified k-fold cross-validation, and select the best-performing model (Logistic Regression, Decision Tree, or Random Forest).
- **Explainable Predictions:** SHAP integration provides visual subject-level contribution scores, avoiding the "black box" automated placement paradigm.
- **PDF Reporting:** Generates printable, A4 guidance reports for academic clinics and parent-teacher meetings.
- **Local Security:** Argon2id password hashing and SQLite foreign-key enforcement protect local application data.

## 🚀 Quickstart Guide

Follow these steps to get the project running on your local machine.

### Prerequisites

- **Python 3.10+** installed on your system.
- Git (to clone the repository).

### 1. Installation

Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/gicheha-titus/final-year-project.git
cd final-year-project

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install the project in development mode
pip install -e .[dev]
```

### 2. Run the Data Pipeline

Before launching the app for the first time, you must run the pipeline. This script will:
1. Generate a synthetic dataset of Grade 7-9 learner assessments.
2. Train and evaluate candidate ML models on the synthetic data.
3. Save the best model bundle (Random Forest by default) to the `artifacts/` folder.
4. Initialize the SQLite database and seed it with the learner records.

```bash
python run_pipeline.py
```
*You should see output indicating the synthetic dataset was generated, models were trained, and the database was initialized.*

### 3. Launch the Application

Once the pipeline has successfully finished, you can launch the PySide6 desktop application:

```bash
# Option 1: Using the provided script
python run_app.py

# Option 2: Running the module directly
python -m cbc_xai.app

# Option 3: Using the installed console script
cbc-xai
```

### 4. Log In

On first launch, the system automatically creates a default administrator account. Use these credentials to log in:

- **Username:** `admin`
- **Password:** `Admin@123`

*(Note: The system will prompt you to change this password immediately upon your first login for security purposes).*

## 🧪 Development & Testing

The project uses `pytest` for the test suite, `ruff` for linting and formatting, and `pre-commit` hooks.

```bash
# Run the test suite
pytest

# Generate a coverage report
pytest --cov=cbc_xai

# Run linters and formatters
ruff check .
ruff format --check .
```

## 📂 Project Structure

- `src/cbc_xai/`: Main application source code.
  - `app.py`: PySide6 UI and main entry point.
  - `modeling.py`: ML pipeline (Model training, evaluation, and SHAP explanations).
  - `storage.py`: SQLite database operations, schema setup, and authentication.
  - `ingestion.py`: Strict CSV parsing and validation logic.
  - `rules.py`: Domain-expert scoring logic mapping subjects to tracks.
  - `domain.py`: Curriculum constants and track-to-pathway mappings.
  - `reporting.py`: PDF report generation using ReportLab.
  - `synthetic_data.py`: Generator for realistic mock assessment data.
- `tests/`: Comprehensive Pytest suite covering core logic and edge cases.
- `data/`: Default location for generated CSV assessment datasets.
- `artifacts/`: Ignored directory where generated models, SHAP charts, PDF reports, and the SQLite database (`cbc_xai.db`) are stored.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
