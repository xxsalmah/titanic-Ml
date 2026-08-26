import pandas as pd


# =====================================
# 1. LOAD DATA
# =====================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)


# =====================================
# 2. LOOK AT THE DATA
# =====================================

print("========== ORIGINAL DATA ==========")

print(df.head())

print("\nShape:", df.shape)


# =====================================
# 3. MISSING VALUES
# =====================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# =====================================
# 4. REMOVE CABIN
# =====================================

df = df.drop(columns=["Cabin"])


# =====================================
# 5. FILL MISSING AGE
# =====================================

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)


# =====================================
# 6. FILL MISSING EMBARKED
# =====================================

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)


# =====================================
# 7. CHECK AGAIN
# =====================================

print("\n========== AFTER CLEANING ==========")

print(df.isnull().sum())


# =====================================
# 8. DISPLAY DATA
# =====================================

print("\n========== CLEAN DATA ==========")

print(df.head())