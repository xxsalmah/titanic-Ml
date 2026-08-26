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
# 2. FEATURE ENGINEERING
# =====================================

# Total number of family members
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# Whether the passenger travelled alone
df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)


# Extract title from name
df["Title"] = (
    df["Name"]
    .str.extract(r",\s*([^.]*)\.")[0]
    .str.strip()
)


# Group uncommon titles
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
    ),

    (
        "scaler",
        StandardScaler()
    )
])


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
# 6. MODEL
# =====================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
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
# 7. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 8. TRAIN
# =====================================

model.fit(
    X_train,
    y_train
)


# =====================================
# 9. PREDICT
# =====================================

predictions = model.predict(
    X_test
)


# =====================================
# 10. RESULT
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== FEATURE ENGINEERING ==========")

print(
    "Accuracy:",
    round(accuracy, 3)
)


# =====================================
# 11. SHOW NEW FEATURES
# =====================================

print("\n========== NEW FEATURES ==========")

print(
    df[
        [
            "Name",
            "FamilySize",
            "IsAlone",
            "Title"
        ]
    ].head(10)
)