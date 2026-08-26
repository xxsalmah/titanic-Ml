import pandas as pd


# =====================================
# 1. LOAD DATA
# =====================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)


# =====================================
# 2. CREATE FAMILY SIZE
# =====================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# =====================================
# 3. CREATE IS ALONE
# =====================================

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


# =====================================
# 4. DISPLAY RESULTS
# =====================================

print("========== NEW FEATURES ==========")

print(
    df[
        [
            "Name",
            "SibSp",
            "Parch",
            "FamilySize",
            "IsAlone"
        ]
    ].head(10)
)
# =====================================
# 5. EXTRACT TITLE
# =====================================

df["Title"] = df["Name"].str.extract(
    r",\s*([^.]*)\."
)[0].str.strip()


# =====================================
# 6. DISPLAY TITLES
# =====================================

print("\n========== TITLES ==========")

print(
    df["Title"].value_counts().head(10)
)