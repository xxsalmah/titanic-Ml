import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("train.csv")


# =====================================
# 2. FEATURES
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
# 3. FEATURE TYPES
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
    ),

    (
        "scaler",
        StandardScaler()
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
# 8. MODELS
# =====================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "SVM": SVC()
}


# =====================================
# 9. TRAIN AND COMPARE
# =====================================

results = {}


for name, classifier in models.items():

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ])


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results[name] = accuracy


# =====================================
# 10. DISPLAY RESULTS
# =====================================

print("========== MODEL COMPARISON ==========")

for name, accuracy in results.items():

    print(
        name,
        ":",
        round(accuracy, 3)
    )


# =====================================
# 11. BEST MODEL
# =====================================

best_model = max(
    results,
    key=results.get
)

best_score = results[best_model]


print("\n========== BEST MODEL ==========")

print(
    "Model:",
    best_model
)

print(
    "Accuracy:",
    round(best_score, 3)
)