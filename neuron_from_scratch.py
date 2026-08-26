import numpy as np


# =====================================
# 1. TRAINING DATA
# =====================================

X = np.array([
    [0],
    [1],
    [2],
    [3],
    [4]
], dtype=float)


y = np.array([
    [0],
    [2],
    [4],
    [6],
    [8]
], dtype=float)


# =====================================
# 2. INITIAL WEIGHT AND BIAS
# =====================================

weight = np.random.randn(1, 1)

bias = np.random.randn(1)


# =====================================
# 3. LEARNING RATE
# =====================================

learning_rate = 0.01


# =====================================
# 4. TRAINING
# =====================================

for epoch in range(1000):

    # -------------------------------
    # FORWARD PASS
    # -------------------------------

    prediction = X @ weight + bias


    # -------------------------------
    # CALCULATE ERROR
    # -------------------------------

    error = prediction - y


    # -------------------------------
    # CALCULATE LOSS
    # -------------------------------

    loss = np.mean(error ** 2)


    # -------------------------------
    # CALCULATE GRADIENTS
    # -------------------------------

    weight_gradient = (
        (2 / len(X)) *
        X.T @ error
    )

    bias_gradient = (
        (2 / len(X)) *
        np.sum(error)
    )


    # -------------------------------
    # UPDATE WEIGHT AND BIAS
    # -------------------------------

    weight -= learning_rate * weight_gradient

    bias -= learning_rate * bias_gradient


# =====================================
# 5. RESULTS
# =====================================

print("========== TRAINING COMPLETE ==========")

print("Weight:", weight)

print("Bias:", bias)

print("Final loss:", loss)


# =====================================
# 6. TEST THE NEURON
# =====================================

test = np.array([
    [5],
    [10],
    [20]
], dtype=float)


predictions = test @ weight + bias


print("\n========== PREDICTIONS ==========")

for value, prediction in zip(
    test.flatten(),
    predictions.flatten()
):

    print(
        "Input:",
        value,
        "→ Prediction:",
        round(prediction, 2)
    )

    .