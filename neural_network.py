from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


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
# 3. SCALE DATA
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =====================================
# 4. CREATE NEURAL NETWORK
# =====================================

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=1000,
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
# 6. PREDICT
# =====================================

predictions = model.predict(
    X_test
)


# =====================================
# 7. ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== NEURAL NETWORK ==========")

print(
    "Accuracy:",
    round(accuracy, 3)
)


# =====================================
# 8. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)