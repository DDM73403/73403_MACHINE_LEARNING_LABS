Lab 4: Data Pipeline and Quality
Daisy Danai Mwale
73403
Group 3

Task 1: Dataset

Dataset: Adult Dataset (OpenML)

Rows: 48842
Columns: 15

The dataset was loaded using fetch_openml() and stored in a pandas DataFrame.

Task 2: Validation

Data types were verified against the expected schema.

Validation Results:
Age Violations: 0
Hours Violations: 0
Capital Gain Violations: 0
Capital Loss Violations: 0

Missing Value Check:
No missing values were found in any column.

Task 3: Cleaning and Transformation

Train/Test Split:
Training Set: (39073, 14)
Test Set: (9769, 14)

Imputation:
Median imputation was applied to 6 numeric columns.

Duplicate Handling:
Duplicate Records Found: 52
Rows Before: 48842
Rows After: 48790
Duplicates Removed: 52

Derived Features:
1. capital_net = capital-gain - capital-loss
2. hours_per_age = hours-per-week / (age - 16)

Task 4: Logistic Regression

Accuracy: 0.8204
F1 Score: 0.5253

Task 5: Label Quality Audit

A random sample of 50 rows was selected for audit.

Audit Sample Size: 50

Task 6: SHA256 Hash

ea800b0a8929e0f778c76354ec30fd41d6210d7627df3f5676ca0436fc2f6271