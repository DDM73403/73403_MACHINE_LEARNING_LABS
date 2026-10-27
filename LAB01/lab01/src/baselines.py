from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_accuracy = accuracy_score(
    y_test,
    lr.predict(X_test)
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_accuracy = accuracy_score(
    y_test,
    rf.predict(X_test)
)

# Save results
results = pd.DataFrame({
    "Model": [
        "LogisticRegression",
        "RandomForest"
    ],
    "Accuracy": [
        round(lr_accuracy, 4),
        round(rf_accuracy, 4)
    ]
})

results.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False
)

print(results)