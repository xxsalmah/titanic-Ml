from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATA
# =====================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================
# 2. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 3. SCALE
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =====================================
# 4. CREATE NETWORK
# =====================================

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42
)


# =====================================
# 5. TRAIN
# =====================================

model.fit(
    X_train,
    y_train
)


# =====================================
# 6. LOSS DURING TRAINING
# =====================================

print("========== TRAINING ==========")

print(
    "Number of iterations:",
    model.n_iter_
)

print(
    "Final loss:",
    round(model.loss_, 4)
)


# =====================================
# 7. LOSS CURVE
# =====================================

plt.figure()

plt.plot(
    model.loss_curve_
)

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.title("Neural Network Learning Curve")

plt.show()