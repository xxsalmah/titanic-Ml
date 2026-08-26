import { useState } from "react";
import "./App.css";

function App() {
  const [form, setForm] = useState({
    Pclass: "3",
    Sex: "male",
    Age: "",
    SibSp: "0",
    Parch: "0",
    Fare: "",
    Embarked: "S",
    Title: "Mr",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // NEW: prediction history
  const [history, setHistory] = useState([]);

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            Pclass: Number(form.Pclass),
            Sex: form.Sex,
            Age: Number(form.Age),
            SibSp: Number(form.SibSp),
            Parch: Number(form.Parch),
            Fare: Number(form.Fare),
            Embarked: form.Embarked,
            Title: form.Title,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResult(data);

      // NEW: add prediction to history
      const newPrediction = {
        id: Date.now(),
        class: form.Pclass,
        sex: form.Sex,
        age: form.Age,
        prediction: data.prediction,
        probability: data.survival_probability,
      };

      setHistory((previous) => [
        newPrediction,
        ...previous,
      ]);

    } catch (error) {
      console.error(error);

      setResult({
        error:
          "Could not connect to the prediction server.",
      });

    } finally {
      setLoading(false);
    }
  };

  // Clear history
  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="app">

      <div className="container">

        {/* HEADER */}

        <div className="header">

          <h1>
            🚢 Titanic Survival Predictor
          </h1>

          <p>
            Machine learning prediction powered by
            Random Forest
          </p>

        </div>


        {/* FORM */}

        <div className="card">

          <form onSubmit={handleSubmit}>

            <div className="form-grid">

              <div className="field">
                <label>
                  Passenger Class
                </label>

                <select
                  name="Pclass"
                  value={form.Pclass}
                  onChange={handleChange}
                >
                  <option value="1">
                    1st Class
                  </option>

                  <option value="2">
                    2nd Class
                  </option>

                  <option value="3">
                    3rd Class
                  </option>
                </select>
              </div>


              <div className="field">
                <label>
                  Sex
                </label>

                <select
                  name="Sex"
                  value={form.Sex}
                  onChange={handleChange}
                >
                  <option value="male">
                    Male
                  </option>

                  <option value="female">
                    Female
                  </option>
                </select>
              </div>


              <div className="field">
                <label>
                  Age
                </label>

                <input
                  type="number"
                  name="Age"
                  value={form.Age}
                  onChange={handleChange}
                  placeholder="e.g. 25"
                  min="0"
                  required
                />
              </div>


              <div className="field">
                <label>
                  Siblings / Spouses
                </label>

                <input
                  type="number"
                  name="SibSp"
                  value={form.SibSp}
                  onChange={handleChange}
                  min="0"
                />
              </div>


              <div className="field">
                <label>
                  Parents / Children
                </label>

                <input
                  type="number"
                  name="Parch"
                  value={form.Parch}
                  onChange={handleChange}
                  min="0"
                />
              </div>


              <div className="field">
                <label>
                  Fare
                </label>

                <input
                  type="number"
                  name="Fare"
                  value={form.Fare}
                  onChange={handleChange}
                  placeholder="e.g. 32.50"
                  min="0"
                  step="0.01"
                  required
                />
              </div>


              <div className="field">
                <label>
                  Embarkation Port
                </label>

                <select
                  name="Embarked"
                  value={form.Embarked}
                  onChange={handleChange}
                >
                  <option value="S">
                    Southampton
                  </option>

                  <option value="C">
                    Cherbourg
                  </option>

                  <option value="Q">
                    Queenstown
                  </option>
                </select>
              </div>


              <div className="field">
                <label>
                  Title
                </label>

                <select
                  name="Title"
                  value={form.Title}
                  onChange={handleChange}
                >
                  <option value="Mr">
                    Mr
                  </option>

                  <option value="Miss">
                    Miss
                  </option>

                  <option value="Mrs">
                    Mrs
                  </option>

                  <option value="Master">
                    Master
                  </option>

                  <option value="Rare">
                    Rare
                  </option>
                </select>
              </div>

            </div>


            <button
              type="submit"
              disabled={loading}
            >
              {loading
                ? "Making Prediction..."
                : "Predict Survival"}
            </button>

          </form>


          {/* RESULT */}

          {result && !result.error && (

            <div className="result">

              <h2>
                {result.prediction}
              </h2>

              <p>
                Estimated survival probability
              </p>

              <div className="probability">
                {result.survival_probability}%
              </div>

            </div>

          )}


          {/* ERROR */}

          {result?.error && (

            <div className="error">
              {result.error}
            </div>

          )}

        </div>


        {/* HISTORY */}

        {history.length > 0 && (

          <div className="history">

            <div className="history-header">

              <h2>
                Prediction History
              </h2>

              <button
                className="clear-button"
                onClick={clearHistory}
              >
                Clear
              </button>

            </div>


            <div className="history-list">

              {history.map((item) => (

                <div
                  className="history-item"
                  key={item.id}
                >

                  <div>
                    <strong>
                      {item.prediction}
                    </strong>

                    <p>
                      {item.sex} ·{" "}
                      {item.age} years ·{" "}
                      Class {item.class}
                    </p>
                  </div>

                  <span>
                    {item.probability}%
                  </span>

                </div>

              ))}

            </div>

          </div>

        )}


        {/* INFORMATION */}

        <div className="info">

          <div className="info-box">
            <h3>🤖 Model</h3>
            <p>Random Forest</p>
          </div>

          <div className="info-box">
            <h3>📊 Features</h3>
            <p>Passenger & family data</p>
          </div>

          <div className="info-box">
            <h3>🧠 Pipeline</h3>
            <p>Preprocessing + ML</p>
          </div>

        </div>

      </div>

    </div>
  );
}

export default App;