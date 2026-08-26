from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


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

model = LinearRegression()


# =====================================
# 4. TRAIN MODEL
# =====================================

model.fit(X_train, y_train)


# =====================================
# 5. MAKE PREDICTIONS
# =====================================

predictions = model.predict(X_test)


# =====================================
# 6. EVALUATE
# =====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# =====================================
# 7. RESULTS
# =====================================

print("========== REGRESSION RESULTS ==========")

print("MAE :", round(mae, 3))

print("MSE :", round(mse, 3))

print("RMSE:", round(rmse, 3))

print("R²  :", round(r2, 3))