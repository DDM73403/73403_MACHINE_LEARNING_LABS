from sklearn.datasets import fetch_openml

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nAge Validation")

age_violations = ((df["age"] < 17) | (df["age"] > 90)).sum()

print("Age Violations:", age_violations)

print("\nHours Validation")

hours_violations = (
        (df["hours-per-week"] < 1) |
        (df["hours-per-week"] > 99)
).sum()

print("Hours Violations:", hours_violations)

print("\nCapital Gain Validation")

capital_gain_violations = (
        (df["capital-gain"] < 0) |
        (df["capital-gain"] > 99999)
).sum()

print("Capital Gain Violations:", capital_gain_violations)

print("\nCapital Loss Validation")

capital_loss_violations = (
        (df["capital-loss"] < 0) |
        (df["capital-loss"] > 4356)
).sum()

print("Capital Loss Violations:", capital_loss_violations)

print("\nMissing Value Check")

for column in df.columns:
    missing_count = (df[column].astype(str) == "?").sum()

    print(column, ":", missing_count)

    from sklearn.model_selection import train_test_split

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set:", X_train.shape)
print("Test Set:", X_test.shape)

from sklearn.impute import SimpleImputer

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

imputer = SimpleImputer(strategy="median")

X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

print("\nImputation Complete")
print("Numeric columns processed:", len(num_cols))

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

print("\nRemoving Duplicates")

df_clean = df.drop_duplicates()

print("Rows Before:", len(df))
print("Rows After:", len(df_clean))
print("Duplicates Removed:", len(df) - len(df_clean))

print("\nCreating Derived Features")

df_clean["capital_net"] = (
        df_clean["capital-gain"]
        - df_clean["capital-loss"]
)

df_clean["hours_per_age"] = (
        df_clean["hours-per-week"]
        / (df_clean["age"] - 16)
)

print("Derived features created:")
print("- capital_net")
print("- hours_per_age")


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df_model = df_clean.copy()

for col in df_model.select_dtypes(include=["category"]).columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("class", axis=1)
y = df_model["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

import joblib

joblib.dump(imputer, "preprocessor.joblib")

print("\nPreprocessor saved as preprocessor.joblib")

sample_rows = df_clean.sample(n=50, random_state=42)

print("\nAudit Sample Size:", len(sample_rows))

import hashlib

dataset_hash = hashlib.sha256(
    df_clean.to_csv(index=False).encode()
).hexdigest()

print("\nSHA256 Hash:")
print(dataset_hash)