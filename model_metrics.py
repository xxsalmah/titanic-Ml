import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("train.csv")


# =====================================
# FEATURE ENGINEERING
# =====================================

# Family size
df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)


# Whether passenger was travelling alone
df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)


# Extract title from passenger name
df["Title"] = (
    df["Name"]
    .str.extract(r",\s*([^.]*)\.", expand=False)
    .str.strip()
)


# Group uncommon titles into Rare
common_titles = [
    "Mr",
    "Miss",
    "Mrs",
    "Master"
]

df["Title"] = df["Title"].apply(
    lambda title:
        title if title in common_titles
        else "Rare"
)


# =====================================
# FEATURES
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
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load(
    "titanic_model.pkl"
)


# =====================================
# MAKE PREDICTIONS
# =====================================

y_pred = model.predict(
    X_test
)


# =====================================
# CALCULATE METRICS
# =====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# =====================================
# DISPLAY PERFORMANCE
# =====================================

print()
print("=====================================")
print("       TITANIC MODEL PERFORMANCE")
print("=====================================")

print()

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)

print()


# =====================================
# CONFUSION MATRIX
# =====================================

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")

print(matrix)

print()


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)