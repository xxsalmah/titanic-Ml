from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib


# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)

CORS(app)


# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "titanic_model.pkl"
)


# =====================================
# HOME ROUTE
# =====================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Titanic Survival Prediction API is running!"
    })


# =====================================
# PREDICTION ROUTE
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json


    # ---------------------------------
    # GET INPUTS
    # ---------------------------------

    pclass = int(data["Pclass"])

    sex = data["Sex"]

    age = float(data["Age"])

    sibsp = int(data["SibSp"])

    parch = int(data["Parch"])

    fare = float(data["Fare"])

    embarked = data["Embarked"]

    title = data["Title"]


    # ---------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------

    family_size = (
        sibsp +
        parch +
        1
    )

    is_alone = int(
        family_size == 1
    )


    # ---------------------------------
    # CREATE DATAFRAME
    # ---------------------------------

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


    # ---------------------------------
    # PREDICTION
    # ---------------------------------

    prediction = model.predict(
        passenger
    )


    # ---------------------------------
    # PROBABILITY
    # ---------------------------------

    probabilities = model.predict_proba(
        passenger
    )

    survival_probability = (
        probabilities[0][1] * 100
    )


    # ---------------------------------
    # RESPONSE
    # ---------------------------------

    if prediction[0] == 1:

        result = "Survived"

    else:

        result = "Did not survive"


    return jsonify({

        "prediction": result,

        "survival_probability":
            round(
                survival_probability,
                2
            )

    })


# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )