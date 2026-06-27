# 🧠 Code Architecture & Execution Flow: The Inside-Out Guide

If the `TECHNICAL_DEEP_DIVE.md` explained the *concepts* behind the project, this document explains the *exact execution* of the code. This is the manual for understanding how the Python files interact, line-by-line, component-by-component. If you want to understand your own code inside out, read this.

---

## 1. Application Startup (`run_app.py` & `app.py`)

When you run `python run_app.py`, the execution begins immediately.

### The Entry Point
1. **`run_app.py`** calls the `main()` function in `src/cbc_xai/app.py`.
2. **`QApplication(sys.argv)`**: This initializes the Qt engine. It binds the Python code to the underlying operating system (Windows/macOS/Linux) so it can draw physical windows on the screen.
3. **The Session Loop**: In `app.py` around line 2360, there is a `start_session()` function. It loops between the `LoginDialog` and the `MainWindow`. 
   - *Why a loop?* If a teacher clicks "Log Out", we don't want the app to crash or completely exit to the desktop. We just destroy the `MainWindow` object and spawn a new `LoginDialog` object. The `while True:` loop makes this possible.

---

## 2. The Database & Authentication (`storage.py`)

Before any UI is drawn, `storage.initialize_database()` is called. 

### The Schema
SQLite creates a file (`artifacts/cbc_xai.db`). The `storage.py` code uses raw SQL `CREATE TABLE IF NOT EXISTS` statements to build two main tables:
1. **`users`**: Stores `username`, `password_hash`, `role`, and `status`.
2. **`assessments`**: Stores the raw grades. Crucially, it has `UNIQUE(learner_id, term, subject)`. 
   - *Why?* This prevents duplicates. A student cannot have two different math grades for "Term 1". The database will literally reject the code if it tries to insert a duplicate.

### The Authentication Flow
When you type a password into the Login UI:
1. `authenticate_user("teacher", "password")` is called.
2. The code fetches the stored `$argon2id...` hash from the database.
3. It passes the plain-text password and the hash to the `argon2.PasswordHasher().verify()` function.
4. Argon2 does the complex math to scramble the input and checks if it matches the stored scramble. If yes, it returns the user dictionary.

---

## 3. Data Ingestion & Validation (`ingestion.py`)

When a teacher uploads a CSV, we do not trust it. The code in `ingestion.py` acts as a bouncer.

### The Validation Steps
1. **Column Check**: It checks if the CSV has the exact required columns (learner_id, subject, score, etc.).
2. **Missing Data**: It drops any row where the `score` or `subject` is literally empty (`pd.isna`).
3. **Type Checking**: It attempts to convert the `score` column to `float`. If a teacher typed "A" instead of "85", it flags the row.
4. **Boundary Checking**: It asserts that every `score` is between `0.0` and `100.0`.
5. **Subject Normalization**: It forces all subject names to match the exact official curriculum names defined in `domain.py`. (e.g., mapping "Maths" to "Mathematics").

If *any* of these fail, it raises an `IngestionError` containing a list of exactly which rows failed and why, which the UI catches and displays to the user.

---

## 4. The Brain: Feature Engineering (`features.py`)

Machine Learning requires fixed-width matrices (a grid of numbers). We must convert "Alice got an 80 in Math in Term 1 and a 90 in Term 2" into a single mathematical row for Alice.

### `build_feature_matrix()`
1. **Group by Learner**: The code uses `pandas.groupby("learner_id")` to isolate one student's history at a time.
2. **Pivot Subjects**: For each subject (e.g., Mathematics), it extracts all the chronological scores.
3. **Calculate 4 Metrics**:
   - **`{subject}_mean`**: `np.mean(scores)` -> The average.
   - **`{subject}_recent_mean`**: `np.mean(scores[-3:])` -> The average of the last 3 terms.
   - **`{subject}_consistency`**: `np.std(scores)` -> The Standard Deviation (how bouncy the grades are).
   - **`{subject}_trend`**: `np.polyfit(x, scores, 1)[0]` -> This fits a straight line through the grades over time. The `[0]` extracts the "slope" of that line. A positive slope means they are improving.
4. **Imputation**: If a student is missing a subject entirely, it fills the mean with `0.0` and the consistency with `100.0` (acting as a penalty).

---

## 5. The Brain: The ML Pipeline (`modeling.py`)

How does the Random Forest actually get trained? Look at `train_and_select_model()`.

### The Pipeline Architecture
The code uses a Scikit-Learn `Pipeline` object:
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier())
])
```
1. **`StandardScaler`**: This takes all the weird numbers (means are 0-100, trends are -5 to +5) and squashes them so they all have a mean of 0 and a variance of 1. ML models perform *much* better when all inputs are on the same scale.
2. **`RandomForestClassifier`**: The actual brain.

### Saving the Brain
Once trained, the `Pipeline` object is saved to the hard drive using `joblib.dump(bundle, MODEL_BUNDLE_PATH)`. This is "Pickling." It takes the live python object in RAM and freezes it into a file (`artifacts/models/selected_model_bundle.joblib`). When the app restarts, it just loads that frozen file back into RAM.

---

## 6. Explainable AI: SHAP Extractor

In `modeling.py`, inside `predict_for_learner()`:
1. The code extracts the pre-trained Random Forest from the bundle.
2. It passes the forest into `shap.TreeExplainer(model)`.
3. It passes the student's feature row into `explainer.shap_values(row)`.
4. **The Magic:** SHAP returns a matrix of impact values. If the model chose "STEM" (Class 0), we look at the SHAP values for Class 0.
   - If `Mathematics_mean` has a SHAP value of `+0.15`, it means math *pushed the decision towards STEM*. This is added to the **Strengths** list.
   - If `Social_Studies_mean` has a SHAP value of `-0.08`, it means social studies *pushed the decision away from STEM*. This is added to the **Limiting Factors** list.

This is how the UI knows exactly what to say in the report!

---

## 7. The UI: Component Breakdown (`app.py`)

If you look at `app.py`, you won't see raw HTML/CSS. You see Python classes inheriting from PySide6 widgets.

### The Main Window (`MainWindow`)
The `MainWindow` inherits from `QMainWindow`. Its core is the `QStackedWidget`.
When you click "Dashboard" on the sidebar, the code calls `self.page_stack.setCurrentIndex(0)`. When you click "User Accounts", it calls `self.page_stack.setCurrentIndex(5)`. The UI doesn't "load" a new page; it just brings an already-existing widget to the front.

### Data Tables (`QTableWidget`)
On the "Cohort Insights" page, the UI uses a `QTableWidget`.
To populate it, the code loops over the pandas DataFrame:
```python
for row_idx, row in data.iterrows():
    self.table.setItem(row_idx, 0, QTableWidgetItem(row["learner_name"]))
```
It physically creates a `QTableWidgetItem` object for every single cell in the grid.

### Threading (Not Used, But Why?)
Notice that when you click "Generate Predictions", the app freezes for a split second. In massive enterprise apps, you would put the ML prediction on a `QThread` (background worker) so the UI doesn't freeze. *However*, because this app is analyzing a single school cohort locally using highly-optimized C-based libraries (numpy/scikit-learn), the prediction takes less than 500ms. Therefore, keeping the code synchronous (running on the main thread) was a deliberate architecture choice to avoid the massive complexity of multithreading in a prototype.

---

## 8. Putting it all together

If you track a single student (Alice) from start to finish:
1. Teacher clicks "Import". `app.py` triggers `ingestion.py`.
2. `ingestion.py` checks Alice's grades, approves them, and calls `storage.py`.
3. `storage.py` INSERTs Alice's grades into SQLite.
4. Teacher goes to Learner Workspace and clicks Alice.
5. `app.py` pulls Alice's grades from `storage.py` and passes them to `modeling.predict_for_learner()`.
6. `modeling.py` calls `features.py` to turn Alice's grades into Math/Science trends.
7. `modeling.py` pushes the trends through the frozen `StandardScaler` and `RandomForestClassifier`.
8. `modeling.py` passes the Random Forest into `shap.TreeExplainer` to figure out *why* Alice got STEM.
9. `modeling.py` calls `rules.py` to see if the human-defined curriculum rules agree with the Random Forest.
10. `app.py` receives the final structured `PredictionResult` object and paints it onto the screen using `QLabel` and `QProgressBar` widgets.
11. If the teacher clicks "Export", `app.py` sends the `PredictionResult` to `reporting.py`, which uses `reportlab.canvas` to physically draw a PDF document and save it to the `artifacts/` folder.

**You now understand every gear, wire, and engine in this codebase.**
