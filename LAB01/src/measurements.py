import time
import statistics

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

    # warmup
    model.fit(X_train, y_train)

    times = []

    for _ in range(5):
        start = time.perf_counter()

        model.fit(X_train, y_train)

        end = time.perf_counter()

        times.append(end - start)

    median_time = statistics.median(times)

    print(f"{name}: {median_time:.6f} seconds")