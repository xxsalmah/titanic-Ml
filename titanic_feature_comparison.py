import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("train.csv")


# =====================================
# 2. CREATE ENGINEERED FEATURES
# =====================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

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
# 3. ORIGINAL FEATURES
# =====================================

original_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]


# =====================================
# 4. ENGINEERED FEATURES
# =====================================

engineered_features = [
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


# =====================================
# 5. TARGET
# =====================================

y = df["Survived"]


# =====================================
# 6. CREATE SAME TRAIN/TEST SPLIT
# =====================================

X_original = df[original_features]

X_engineered = df[engineered_features]


X_original_train, X_original_test, y_train, y_test = train_test_split(
    X_original,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X_engineered_train, X_engineered_test, _, _ = train_test_split(
    X_engineered,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 7. PREPROCESSOR FOR ORIGINAL DATA
# =====================================

original_numeric = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

original_categorical = [
    "Sex",
    "Embarked"
]


original_numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


original_categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


original_preprocessor = ColumnTransformer([
    (
        "numeric",
        original_numeric_pipeline,
        original_numeric
    ),
    (
        "categorical",
        original_categorical_pipeline,
        original_categorical
    )
])


# =====================================
# 8. PREPROCESSOR FOR ENGINEERED DATA
# =====================================

engineered_numeric = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]

engineered_categorical = [
    "Sex",
    "Embarked",
    "Title"
]


engineered_numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


engineered_categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


engineered_preprocessor = ColumnTransformer([
    (
        "numeric",
        engineered_numeric_pipeline,
        engineered_numeric
    ),
    (
        "categorical",
        engineered_categorical_pipeline,
        engineered_categorical
    )
])


# =====================================
# 9. ORIGINAL MODEL
# =====================================

original_model = Pipeline([
    (
        "preprocessor",
        original_preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
])


# =====================================
# 10. ENGINEERED MODEL
# =====================================

engineered_model = Pipeline([
    (
        "preprocessor",
        engineered_preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
])


# =====================================
# 11. TRAIN ORIGINAL MODEL
# =====================================

original_model.fit(
    X_original_train,
    y_train
)


# =====================================
# 12. TRAIN ENGINEERED MODEL
# =====================================

engineered_model.fit(
    X_engineered_train,
    y_train
)


# =====================================
# 13. PREDICTIONS
# =====================================

original_predictions = original_model.predict(
    X_original_test
)

engineered_predictions = engineered_model.predict(
    X_engineered_test
)


# =====================================
# 14. ACCURACY
# =====================================

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

engineered_accuracy = accuracy_score(
    y_test,
    engineered_predictions
)


# =====================================
# 15. RESULTS
# =====================================

print("========== FEATURE COMPARISON ==========")

print(
    "Original Features:",
    round(original_accuracy, 3)
)

print(
    "Engineered Features:",
    round(engineered_accuracy, 3)
)


# =====================================
# 16. IMPROVEMENT
# =====================================

improvement = (
    engineered_accuracy -
    original_accuracy
)


print(
    "Improvement:",
    round(improvement, 3)
)