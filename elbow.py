from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATA
# =====================================

iris = load_iris()

X = iris.data


# =====================================
# 2. SCALE DATA
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =====================================
# 3. TEST DIFFERENT NUMBERS OF CLUSTERS
# =====================================

inertias = []

k_values = range(1, 11)


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)


# =====================================
# 4. PRINT RESULTS
# =====================================

print("========== INERTIA ==========")

for k, inertia in zip(k_values, inertias):

    print(
        "K =", k,
        "Inertia =", round(inertia, 2)
    )


# =====================================
# 5. PLOT ELBOW
# =====================================

plt.figure()

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.xticks(
    range(1, 11)
)

plt.show()