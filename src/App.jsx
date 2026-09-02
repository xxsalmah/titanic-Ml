import { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
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

  const [history, setHistory] = useState([]);

  const [stats, setStats] = useState({
    total_predictions: 0,
    survived: 0,
    did_not_survive: 0,
    average_probability: 0,
    survival_rate: 0,
  });

  const [metrics, setMetrics] = useState({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1_score: 0,
    confusion_matrix: [
      [0, 0],
      [0, 0],
    ],
  });


  // =====================================
  // CHART DATA
  // =====================================

  const chartData = [
    {
      name: "Survived",
      value: stats.survived,
    },
    {
      name: "Did Not Survive",
      value: stats.did_not_survive,
    },
  ];


  // =====================================
  // HANDLE FORM CHANGES
  // =====================================

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };


  // =====================================
  // LOAD HISTORY
  // =====================================

  const loadHistory = async () => {
    try {
      const response = await fetch(
        "https://titanic-survival-api-xf52.onrender.com/history"
      );

      if (!response.ok) {
        throw new Error("Failed to load history");
      }

      const data = await response.json();

      setHistory(data);

    } catch (error) {
      console.error(
        "Error loading history:",
        error
      );
    }
  };


  // =====================================
  // LOAD STATISTICS
  // =====================================

  const loadStats = async () => {
    try {
      const response = await fetch(
        "https://titanic-survival-api-xf52.onrender.com/stats"
      );

      if (!response.ok) {
        throw new Error(
          "Failed to load statistics"
        );
      }

      const data = await response.json();

      setStats(data);

    } catch (error) {
      console.error(
        "Error loading statistics:",
        error
      );
    }
  };


  // =====================================
  // LOAD MODEL METRICS
  // =====================================

  const loadMetrics = async () => {
    try {
      const response = await fetch(
        "https://titanic-survival-api-xf52.onrender.com/metrics"
      );

      if (!response.ok) {
        throw new Error(
          "Failed to load metrics"
        );
      }

      const data = await response.json();

      setMetrics(data);

    } catch (error) {
      console.error(
        "Error loading metrics:",
        error
      );
    }
  };


  // =====================================
  // LOAD DATA WHEN PAGE OPENS
  // =====================================

  useEffect(() => {
    loadHistory();
    loadStats();
    loadMetrics();
  }, []);


  // =====================================
  // MAKE PREDICTION
  // =====================================

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setResult(null);

    try {

      const response = await fetch(
        "https://titanic-survival-api-xf52.onrender.com/predict",
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
        throw new Error(
          "Prediction request failed."
        );
      }


      const data =
        await response.json();


      setResult(data);


      // Reload database information

      await loadHistory();

      await loadStats();


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


  // =====================================
  // CLEAR HISTORY
  // =====================================

  const clearHistory = async () => {

    try {

      const response = await fetch(
        "https://titanic-survival-api-xf52.onrender.com/history",
        {
          method: "DELETE",
        }
      );


      if (!response.ok) {
        throw new Error(
          "Failed to clear history"
        );
      }


      setHistory([]);

      setStats({
        total_predictions: 0,
        survived: 0,
        did_not_survive: 0,
        average_probability: 0,
        survival_rate: 0,
      });

      setResult(null);


    } catch (error) {

      console.error(
        "Error clearing history:",
        error
      );

    }
  };


  // =====================================
  // UI
  // =====================================

  return (
    <div className="app">

      <div className="container">


        {/* =================================
            HEADER
        ================================= */}

        <div className="header">

          <h1>
            🚢 Titanic Survival Predictor
          </h1>

          <p>
            Machine learning prediction powered
            by Random Forest
          </p>

        </div>


        {/* =================================
            STATISTICS
        ================================= */}

        <div className="stats-grid">

          <div className="stat-card">

            <span>
              Total Predictions
            </span>

            <strong>
              {stats.total_predictions}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              Survived
            </span>

            <strong>
              {stats.survived}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              Did Not Survive
            </span>

            <strong>
              {stats.did_not_survive}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              Survival Rate
            </span>

            <strong>
              {stats.survival_rate}%
            </strong>

          </div>

        </div>


        {/* =================================
            PREDICTION FORM
        ================================= */}

        <div className="card">

          <form onSubmit={handleSubmit}>

            <div className="form-grid">


              {/* PASSENGER CLASS */}

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


              {/* SEX */}

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


              {/* AGE */}

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


              {/* SIBSP */}

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


              {/* PARCH */}

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


              {/* FARE */}

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


              {/* EMBARKED */}

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


              {/* TITLE */}

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


            {/* PREDICT BUTTON */}

            <button
              type="submit"
              disabled={loading}
            >

              {loading
                ? "Making Prediction..."
                : "Predict Survival"}

            </button>

          </form>


          {/* =================================
              RESULT
          ================================= */}

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


        {/* =================================
            MODEL PERFORMANCE
        ================================= */}

        <div className="performance-card">

          <div className="performance-header">

            <h2>
              🧠 Model Performance
            </h2>

            <p>
              Random Forest evaluation metrics
            </p>

          </div>


          <div className="metrics-grid">

            <div className="metric-box">

              <span>
                Accuracy
              </span>

              <strong>
                {metrics.accuracy}%
              </strong>

            </div>


            <div className="metric-box">

              <span>
                Precision
              </span>

              <strong>
                {metrics.precision}%
              </strong>

            </div>


            <div className="metric-box">

              <span>
                Recall
              </span>

              <strong>
                {metrics.recall}%
              </strong>

            </div>


            <div className="metric-box">

              <span>
                F1 Score
              </span>

              <strong>
                {metrics.f1_score}%
              </strong>

            </div>

          </div>


          {/* CONFUSION MATRIX */}

          <div className="confusion-section">

            <h3>
              Confusion Matrix
            </h3>

            <div className="confusion-matrix">

              <div></div>

              <div className="matrix-label">
                Predicted 0
              </div>

              <div className="matrix-label">
                Predicted 1
              </div>


              <div className="matrix-label">
                Actual 0
              </div>

              <div className="matrix-value">
                {metrics.confusion_matrix[0][0]}
              </div>

              <div className="matrix-value">
                {metrics.confusion_matrix[0][1]}
              </div>


              <div className="matrix-label">
                Actual 1
              </div>

              <div className="matrix-value">
                {metrics.confusion_matrix[1][0]}
              </div>

              <div className="matrix-value">
                {metrics.confusion_matrix[1][1]}
              </div>

            </div>

          </div>

        </div>


        {/* =================================
            SURVIVAL CHART
        ================================= */}

        <div className="chart-card">

          <div className="chart-header">

            <h2>
              📈 Survival Distribution
            </h2>

            <p>
              Breakdown of prediction results
            </p>

          </div>


          {stats.total_predictions > 0 ? (

            <div className="chart-container">

              <ResponsiveContainer
                width="100%"
                height={320}
              >

                <PieChart>

                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    outerRadius={105}
                    dataKey="value"
                    label={({
                      name,
                      percent,
                    }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                  >

                    <Cell fill="#557653" />

                    <Cell fill="#8a6652" />

                  </Pie>


                  <Tooltip />

                  <Legend />

                </PieChart>

              </ResponsiveContainer>

            </div>

          ) : (

            <div className="empty-chart">

              <p>
                Make a prediction to see your
                chart.
              </p>

            </div>

          )}

        </div>


        {/* =================================
            HISTORY
        ================================= */}

        {history.length > 0 && (

          <div className="history">

            <div className="history-header">

              <h2>
                Prediction History
              </h2>

              <button
                type="button"
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
                      {item.Sex} ·{" "}
                      {item.Age} years ·{" "}
                      Class {item.Pclass}
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


        {/* =================================
            INFORMATION
        ================================= */}

        <div className="info">

          <div className="info-box">

            <h3>
              🤖 Model
            </h3>

            <p>
              Random Forest
            </p>

          </div>


          <div className="info-box">

            <h3>
              📊 Features
            </h3>

            <p>
              Passenger & family data
            </p>

          </div>


          <div className="info-box">

            <h3>
              🧠 Pipeline
            </h3>

            <p>
              Preprocessing + ML
            </p>

          </div>

        </div>


      </div>

    </div>
  );
}

export default App;

