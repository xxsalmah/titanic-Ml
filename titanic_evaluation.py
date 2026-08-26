import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATA
# =====================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)


# =====================================
# 2. FEATURES AND TARGET
# =====================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]

y = df["Survived"]


# =====================================
# 3. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 4. COLUMN TYPES
# =====================================

numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Pclass"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# =====================================
# 5. NUMERIC PIPELINE
# =====================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# =====================================
# 6. CATEGORICAL PIPELINE
# =====================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# =====================================
# 7. PREPROCESSOR
# =====================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# =====================================
# 8. MODEL PIPELINE
# =====================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# =====================================
# 9. TRAIN
# =====================================

model.fit(X_train, y_train)


# =====================================
# 10. PREDICTIONS
# =====================================

predictions = model.predict(X_test)


# =====================================
# 11. METRICS
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print("========== MODEL EVALUATION ==========")

print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))


# =====================================
# 12. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions
    )
)


# =====================================
# 13. CONFUSION MATRIX
# =====================================

matrix = confusion_matrix(
    y_test,
    predictions
)

print("\n========== CONFUSION MATRIX ==========")

print(matrix)


# =====================================
# 14. VISUALIZE CONFUSION MATRIX
# =====================================

plt.figure()

plt.imshow(matrix)

plt.title("Titanic Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.yticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.colorbar()

plt.show()