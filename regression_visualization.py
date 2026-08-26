from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATA
# =====================================

housing = fetch_california_housing()

X = housing.data
y = housing.target


# =====================================
# 2. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# 3. CREATE MODEL
# =====================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# =====================================
# 4. TRAIN
# =====================================

model.fit(X_train, y_train)


# =====================================
# 5. PREDICT
# =====================================

predictions = model.predict(X_test)


# =====================================
# 6. METRICS
# =====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("========== RESULTS ==========")

print("MAE:", round(mae, 3))

print("R²:", round(r2, 3))


# =====================================
# 7. ACTUAL VS PREDICTED
# =====================================

plt.figure()

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted House Values")

plt.show()


# =====================================
# 8. CALCULATE ERRORS
# =====================================

errors = y_test - predictions


# =====================================
# 9. ERROR PLOT
# =====================================

plt.figure()

plt.scatter(
    predictions,
    errors,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")

plt.ylabel("Prediction Error")

plt.title("Prediction Errors")

plt.show()