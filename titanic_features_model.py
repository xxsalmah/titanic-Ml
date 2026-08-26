import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# =====================================
# 1. LOAD DATA
# =====================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)


# =====================================
# 2. FEATURE ENGINEERING
# =====================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df["Title"] = (
    df["Name"]
    .str.extract(r",\s*([^.]*)\.")[0]
    .str.strip()
)


# =====================================
# 3. SELECT FEATURES
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
# 4. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 5. DEFINE COLUMN TYPES
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
# 6. NUMERIC PIPELINE
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
# 7. CATEGORICAL PIPELINE
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
# 8. PREPROCESSOR
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
# 9. COMPLETE PIPELINE
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
# 10. TRAIN
# =====================================

model.fit(X_train, y_train)


# =====================================
# 11. PREDICT
# =====================================

predictions = model.predict(X_test)


# =====================================
# 12. TEST ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("========== TEST RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy, 3)
)


# =====================================
# 13. CROSS-VALIDATION
# =====================================

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


print("\n========== CROSS-VALIDATION ==========")

print("Scores:")

for score in scores:
    print(round(score, 3))


print(
    "\nAverage:",
    round(scores.mean(), 3)
)