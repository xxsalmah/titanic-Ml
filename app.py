from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

CORS(app)


# =====================================
# DATABASE MODEL
# =====================================

class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    pclass = db.Column(db.Integer)

    sex = db.Column(
        db.String(20)
    )

    age = db.Column(
        db.Float
    )

    sibsp = db.Column(
        db.Integer
    )

    parch = db.Column(
        db.Integer
    )

    fare = db.Column(
        db.Float
    )

    embarked = db.Column(
        db.String(5)
    )

    title = db.Column(
        db.String(20)
    )

    prediction = db.Column(
        db.String(50)
    )

    probability = db.Column(
        db.Float
    )


# =====================================
# CREATE DATABASE
# =====================================

with app.app_context():

    db.create_all()


# =====================================
# LOAD MACHINE LEARNING MODEL
# =====================================

model = joblib.load(
    "titanic_model.pkl"
)


# =====================================
# HOME ROUTE
# =====================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({
        "message":
            "Titanic Survival Prediction API is running!"
    })


# =====================================
# PREDICTION ROUTE
# =====================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.json


        # ---------------------------------
        # GET INPUTS
        # ---------------------------------

        pclass = int(
            data["Pclass"]
        )

        sex = data["Sex"]

        age = float(
            data["Age"]
        )

        sibsp = int(
            data["SibSp"]
        )

        parch = int(
            data["Parch"]
        )

        fare = float(
            data["Fare"]
        )

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
        # CREATE PASSENGER DATAFRAME
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
        # MAKE PREDICTION
        # ---------------------------------

        prediction = model.predict(
            passenger
        )


        # ---------------------------------
        # SURVIVAL PROBABILITY
        # ---------------------------------

        probabilities = model.predict_proba(
            passenger
        )

        survival_probability = (
            probabilities[0][1] * 100
        )


        # ---------------------------------
        # CONVERT RESULT TO TEXT
        # ---------------------------------

        if prediction[0] == 1:

            result = "Survived"

        else:

            result = "Did not survive"


        # =================================
        # SAVE TO DATABASE
        # =================================

        new_prediction = Prediction(

            pclass=pclass,

            sex=sex,

            age=age,

            sibsp=sibsp,

            parch=parch,

            fare=fare,

            embarked=embarked,

            title=title,

            prediction=result,

            probability=round(
                survival_probability,
                2
            )
        )


        db.session.add(
            new_prediction
        )

        db.session.commit()


        # =================================
        # RETURN RESULT
        # =================================

        return jsonify({

            "prediction":
                result,

            "survival_probability":
                round(
                    survival_probability,
                    2
                )

        })


    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return jsonify({

            "error":
                "Could not make prediction."

        }), 500


# =====================================
# HISTORY ROUTE
# =====================================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    predictions = Prediction.query.order_by(
        Prediction.id.desc()
    ).all()


    history_data = []


    for item in predictions:

        history_data.append({

            "id":
                item.id,

            "Pclass":
                item.pclass,

            "Sex":
                item.sex,

            "Age":
                item.age,

            "SibSp":
                item.sibsp,

            "Parch":
                item.parch,

            "Fare":
                item.fare,

            "Embarked":
                item.embarked,

            "Title":
                item.title,

            "prediction":
                item.prediction,

            "probability":
                item.probability

        })


    return jsonify(
        history_data
    )


# =====================================
# CLEAR HISTORY
# =====================================

@app.route(
    "/history",
    methods=["DELETE"]
)
def clear_history():

    try:

        Prediction.query.delete()

        db.session.commit()

        return jsonify({

            "message":
                "Prediction history cleared."

        })


    except Exception as error:

        db.session.rollback()

        print(
            "Clear history error:",
            error
        )

        return jsonify({

            "error":
                "Could not clear history."

        }), 500


# =====================================
# STATISTICS ROUTE
# =====================================

@app.route(
    "/stats",
    methods=["GET"]
)
def stats():

    predictions = Prediction.query.all()


    total_predictions = len(
        predictions
    )


    survived = sum(
        1
        for item in predictions
        if item.prediction == "Survived"
    )


    did_not_survive = sum(
        1
        for item in predictions
        if item.prediction == "Did not survive"
    )


    if total_predictions > 0:

        average_probability = sum(
            item.probability
            for item in predictions
        ) / total_predictions

        survival_rate = (
            survived /
            total_predictions
        ) * 100

    else:

        average_probability = 0

        survival_rate = 0


    return jsonify({

        "total_predictions":
            total_predictions,

        "survived":
            survived,

        "did_not_survive":
            did_not_survive,

        "average_probability":
            round(
                average_probability,
                2
            ),

        "survival_rate":
            round(
                survival_rate,
                2
            )

    })


# =====================================
# MODEL PERFORMANCE ROUTE
# =====================================

@app.route(
    "/metrics",
    methods=["GET"]
)
def metrics():

    try:

        # ---------------------------------
        # LOAD TITANIC DATASET
        # ---------------------------------

        df = pd.read_csv(
            "train.csv"
        )


        # ---------------------------------
        # FEATURE ENGINEERING
        # ---------------------------------

        df["FamilySize"] = (
            df["SibSp"] +
            df["Parch"] +
            1
        )


        df["IsAlone"] = (
            df["FamilySize"] == 1
        ).astype(int)


        # Extract passenger title
        df["Title"] = (
            df["Name"]
            .str.extract(
                r",\s*([^.]*)\.",
                expand=False
            )
            .str.strip()
        )


        # ---------------------------------
        # GROUP RARE TITLES
        # ---------------------------------

        common_titles = [
            "Mr",
            "Miss",
            "Mrs",
            "Master"
        ]


        df["Title"] = df["Title"].apply(

            lambda title:
                title
                if title in common_titles
                else "Rare"

        )


        # ---------------------------------
        # FEATURES
        # ---------------------------------

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


        # ---------------------------------
        # TRAIN / TEST SPLIT
        # ---------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.2,

                random_state=42,

                stratify=y

            )
        )


        # ---------------------------------
        # MAKE PREDICTIONS
        # ---------------------------------

        y_pred = model.predict(
            X_test
        )


        # ---------------------------------
        # CALCULATE METRICS
        # ---------------------------------

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


        # ---------------------------------
        # CONFUSION MATRIX
        # ---------------------------------

        matrix = confusion_matrix(
            y_test,
            y_pred
        )


        # ---------------------------------
        # RETURN METRICS
        # ---------------------------------

        return jsonify({

            "accuracy":
                round(
                    accuracy * 100,
                    2
                ),

            "precision":
                round(
                    precision * 100,
                    2
                ),

            "recall":
                round(
                    recall * 100,
                    2
                ),

            "f1_score":
                round(
                    f1 * 100,
                    2
                ),

            "confusion_matrix":
                matrix.tolist()

        })


    except Exception as error:

        print(
            "Metrics error:",
            error
        )

        return jsonify({

            "error":
                "Could not calculate model metrics."

        }), 500


# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":

    app.run(

        debug=True,

        port=5000

    )

