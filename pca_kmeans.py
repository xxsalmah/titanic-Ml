from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# 3. REDUCE DIMENSIONS WITH PCA
# =====================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# =====================================
# 4. K-MEANS
# =====================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)


# =====================================
# 5. SILHOUETTE SCORE
# =====================================

score = silhouette_score(
    X_pca,
    clusters
)


print("========== PCA + K-MEANS ==========")

print(
    "Silhouette Score:",
    round(score, 3)
)


# =====================================
# 6. CLUSTER COUNTS
# =====================================

print("\n========== CLUSTER COUNTS ==========")

for cluster in range(3):

    count = (clusters == cluster).sum()

    print(
        "Cluster",
        cluster,
        ":",
        count,
        "samples"
    )


# =====================================
# 7. VISUALIZE CLUSTERS
# =====================================

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA + K-Means Clustering")

plt.colorbar(
    label="Cluster"
)

plt.show()