import pandas as pd
import joblib


# =====================================
# 1. LOAD SAVED MODEL
# =====================================

model = joblib.load(
    "titanic_model.pkl"
)


print("========== TITANIC SURVIVAL PREDICTOR ==========")


# =====================================
# 2. GET PASSENGER INFORMATION
# =====================================

pclass = int(
    input("Passenger class (1, 2, or 3): ")
)

sex = input(
    "Sex (male/female): "
).lower()

age = float(
    input("Age: ")
)

sibsp = int(
    input("Number of siblings/spouses aboard: ")
)

parch = int(
    input("Number of parents/children aboard: ")
)

fare = float(
    input("Fare: ")
)

embarked = input(
    "Embarked (C, Q, or S): "
).upper()


# =====================================
# 3. FEATURE ENGINEERING
# =====================================

family_size = (
    sibsp +
    parch +
    1
)

is_alone = int(
    family_size == 1
)


# =====================================
# 4. TITLE
# =====================================

if sex == "male":

    title = "Mr"

else:

    title = input(
        "Title (Miss/Mrs): "
    ).strip()

    if title not in ["Miss", "Mrs"]:

        title = "Rare"


# =====================================
# 5. CREATE DATAFRAME
# =====================================

passenger = pd.DataFrame([
    {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": title
    }
])


# =====================================
# 6. MAKE PREDICTION
# =====================================

prediction = model.predict(
    passenger
)


# =====================================
# 7. DISPLAY RESULT
# =====================================

print("\n========== RESULT ==========")

if prediction[0] == 1:

    print(
        "Prediction: SURVIVED 🚢"
    )

else:

    print(
        "Prediction: DID NOT SURVIVE"
    )


# =====================================
# 8. PROBABILITY
# =====================================

probabilities = model.predict_proba(
    passenger
)

survival_probability = (
    probabilities[0][1] * 100
)

print(
    "Survival probability:",
    round(survival_probability, 2),
    "%"
)