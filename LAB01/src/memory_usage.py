import os
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

process = psutil.Process(os.getpid())

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

models = {
    "LogisticRegression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

for name, model in models.items():

    before = process.memory_info().rss

    model.fit(X_train, y_train)

    after = process.memory_info().rss

    memory_mb = (after - before) / (1024 * 1024)

    print(
        f"{name}: {memory_mb:.2f} MB"
    )