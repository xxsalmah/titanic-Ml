from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
# 3. TEST DIFFERENT K VALUES
# =====================================

k_values = range(2, 11)

scores = []


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        clusters
    )

    scores.append(score)


# =====================================
# 4. PRINT SCORES
# =====================================

print("========== SILHOUETTE SCORES ==========")

for k, score in zip(k_values, scores):

    print(
        "K =",
        k,
        "Score =",
        round(score, 3)
    )


# =====================================
# 5. FIND BEST K
# =====================================

best_index = scores.index(max(scores))

best_k = list(k_values)[best_index]

best_score = scores[best_index]


print("\n========== BEST RESULT ==========")

print("Best K:", best_k)

print("Best Score:", round(best_score, 3))


# =====================================
# 6. PLOT
# =====================================

plt.figure()

plt.plot(
    k_values,
    scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.title("Silhouette Score")

plt.xticks(
    list(k_values)
)

plt.show()