import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier


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
# 5. PREPROCESSING
# =====================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

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
# 6. MODEL PIPELINE
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
# 7. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 8. PARAMETERS
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
# 9. GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


print("========== TRAINING ==========")

grid_search.fit(
    X_train,
    y_train
)


# =====================================
# 10. BEST MODEL
# =====================================

best_model = grid_search.best_estimator_


print("\n========== BEST PARAMETERS ==========")

print(
    grid_search.best_params_
)


print("\n========== BEST CV SCORE ==========")

print(
    round(
        grid_search.best_score_,
        3
    )
)


# =====================================
# 11. SAVE MODEL
# =====================================

joblib.dump(
    best_model,
    "titanic_model.pkl"
)


print("\n========== MODEL SAVED ==========")

print(
    "Saved as: titanic_model.pkl"
)