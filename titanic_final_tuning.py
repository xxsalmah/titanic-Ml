import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("train.csv")


# =====================================
# 2. FEATURE ENGINEERING
# =====================================

df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)


df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)


df["Title"] = (
    df["Name"]
    .str.extract(r",\s*([^.]*)\.")[0]
    .str.strip()
)


common_titles = [
    "Mr",
    "Miss",
    "Mrs",
    "Master"
]


df["Title"] = df["Title"].where(
    df["Title"].isin(common_titles),
    "Rare"
)


# =====================================
# 3. FEATURES
# =====================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title"
]


X = df[features]

y = df["Survived"]


# =====================================
# 4. FEATURE TYPES
# =====================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]


categorical_features = [
    "Sex",
    "Embarked",
    "Title"
]


# =====================================
# 5. NUMERIC PIPELINE
# =====================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
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
        OneHotEncoder(
            handle_unknown="ignore"
        )
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
# 8. COMPLETE PIPELINE
# =====================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42
        )
    )
])


# =====================================
# 9. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 10. PARAMETERS TO SEARCH
# =====================================

param_grid = {

    "classifier__n_estimators": [
        100,
        200,
        300
    ],

    "classifier__max_depth": [
        None,
        5,
        10,
        15
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        4
    ]

}


# =====================================
# 11. GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# =====================================
# 12. TRAIN GRID SEARCH
# =====================================

print("========== STARTING GRID SEARCH ==========")

grid_search.fit(
    X_train,
    y_train
)


# =====================================
# 13. BEST PARAMETERS
# =====================================

print("\n========== BEST PARAMETERS ==========")

print(
    grid_search.best_params_
)


# =====================================
# 14. BEST CV SCORE
# =====================================

print("\n========== BEST CV SCORE ==========")

print(
    round(
        grid_search.best_score_,
        3
    )
)


# =====================================
# 15. TEST SET
# =====================================

predictions = grid_search.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========== TEST RESULTS ==========")

print(
    "Test Accuracy:",
    round(test_accuracy, 3)
)


# =====================================
# 16. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Did Not Survive",
            "Survived"
        ]
    )
)