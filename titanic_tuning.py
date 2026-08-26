import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


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
# 8. COMPLETE PIPELINE
# =====================================

pipeline = Pipeline([
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
# 9. PARAMETERS TO TEST
# =====================================

parameters = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


# =====================================
# 10. GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="accuracy"
)


# =====================================
# 11. TRAIN
# =====================================

print("Training model...")

grid_search.fit(X_train, y_train)


# =====================================
# 12. BEST PARAMETERS
# =====================================

print("\n========== BEST PARAMETERS ==========")

print(grid_search.best_params_)


# =====================================
# 13. BEST CV SCORE
# =====================================

print("\n========== BEST CV SCORE ==========")

print(
    round(
        grid_search.best_score_,
        3
    )
)


# =====================================
# 14. FINAL TEST
# =====================================

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)


# =====================================
# 15. TEST ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========== TEST RESULTS ==========")

print(
    "Test accuracy:",
    round(accuracy, 3)
)


# =====================================
# 16. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions
    )
)