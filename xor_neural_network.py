import numpy as np


# =====================================
# 1. DATA
# =====================================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)


y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# =====================================
# 2. ACTIVATION FUNCTIONS
# =====================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# =====================================
# 3. INITIALIZE WEIGHTS
# =====================================

np.random.seed(42)

W1 = np.random.randn(2, 4)

b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)

b2 = np.zeros((1, 1))


# =====================================
# 4. LEARNING RATE
# =====================================

learning_rate = 0.5


# =====================================
# 5. TRAINING
# =====================================

for epoch in range(10000):

    # -------------------------------
    # FORWARD PROPAGATION
    # -------------------------------

    z1 = X @ W1 + b1

    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2

    output = sigmoid(z2)


    # -------------------------------
    # ERROR
    # -------------------------------

    error = output - y


    # -------------------------------
    # BACKPROPAGATION
    # -------------------------------

    output_gradient = (
        error *
        sigmoid_derivative(output)
    )

    W2_gradient = a1.T @ output_gradient

    b2_gradient = np.sum(
        output_gradient,
        axis=0,
        keepdims=True
    )


    hidden_gradient = (
        output_gradient @ W2.T
        *
        sigmoid_derivative(a1)
    )

    W1_gradient = X.T @ hidden_gradient

    b1_gradient = np.sum(
        hidden_gradient,
        axis=0,
        keepdims=True
    )


    # -------------------------------
    # UPDATE WEIGHTS
    # -------------------------------

    W2 -= learning_rate * W2_gradient

    b2 -= learning_rate * b2_gradient

    W1 -= learning_rate * W1_gradient

    b1 -= learning_rate * b1_gradient


# =====================================
# 6. FINAL PREDICTIONS
# =====================================

print("========== XOR PREDICTIONS ==========")

for inputs, prediction in zip(X, output):

    print(
        inputs,
        "→",
        round(float(prediction[0]), 3)
    )