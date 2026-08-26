import pandas as pd

from sklearn.model_selection import train_test_split

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
# 3. TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# 4. DEFINE COLUMN TYPES
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
# 7. COLUMN TRANSFORMER
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
# 8. COMPLETE ML PIPELINE
# =====================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression()
    )
])


# =====================================
# 9. TRAIN
# =====================================

model.fit(X_train, y_train)


# =====================================
# 10. PREDICT
# =====================================

predictions = model.predict(X_test)


# =====================================
# 11. EVALUATE
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== TITANIC MODEL ==========")

print("Accuracy:", round(accuracy, 2))