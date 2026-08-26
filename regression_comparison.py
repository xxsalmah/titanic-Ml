from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score


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
# 3. LINEAR REGRESSION PIPELINE
# =====================================

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# =====================================
# 4. RANDOM FOREST MODEL
# =====================================

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# =====================================
# 5. TRAIN LINEAR REGRESSION
# =====================================

linear_model.fit(
    X_train,
    y_train
)


# =====================================
# 6. TRAIN RANDOM FOREST
# =====================================

forest_model.fit(
    X_train,
    y_train
)


# =====================================
# 7. PREDICTIONS
# =====================================

linear_predictions = linear_model.predict(
    X_test
)

forest_predictions = forest_model.predict(
    X_test
)


# =====================================
# 8. LINEAR REGRESSION RESULTS
# =====================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# =====================================
# 9. RANDOM FOREST RESULTS
# =====================================

forest_mae = mean_absolute_error(
    y_test,
    forest_predictions
)

forest_r2 = r2_score(
    y_test,
    forest_predictions
)


# =====================================
# 10. RESULTS
# =====================================

print("========== MODEL COMPARISON ==========")

print("\nLinear Regression")

print(
    "MAE:",
    round(linear_mae, 3)
)

print(
    "R²:",
    round(linear_r2, 3)
)


print("\nRandom Forest")

print(
    "MAE:",
    round(forest_mae, 3)
)

print(
    "R²:",
    round(forest_r2, 3)
)