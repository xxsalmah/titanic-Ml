import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("train.csv")


# =====================================
# 2. SELECT FEATURES
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
# 3. DEFINE FEATURES
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
# 4. NUMERIC PIPELINE
# =====================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# =====================================
# 5. CATEGORICAL PIPELINE
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
# 6. PREPROCESSOR
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
# 7. COMPLETE PIPELINE
# =====================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])


# =====================================
# 8. TRAIN/TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 9. TRAIN
# =====================================

model.fit(
    X_train,
    y_train
)


# =====================================
# 10. TEST SET
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== TEST SET ==========")

print(
    "Accuracy:",
    round(accuracy, 3)
)


# =====================================
# 11. CROSS-VALIDATION
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


# =====================================
# 12. RESULTS
# =====================================

print("\n========== CROSS-VALIDATION ==========")

print("Scores:")

for score in scores:

    print(
        round(score, 3)
    )


print(
    "\nMean Accuracy:",
    round(scores.mean(), 3)
)


print(
    "Standard Deviation:",
    round(scores.std(), 3)
)