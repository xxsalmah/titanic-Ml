import numpy as np

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

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
# 3. CREATE K-MEANS MODEL
# =====================================

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# =====================================
# 4. TRAIN
# =====================================

model.fit(X_scaled)


# =====================================
# 5. GET CLUSTERS
# =====================================

clusters = model.labels_


# =====================================
# 6. DISPLAY RESULTS
# =====================================

print("========== CLUSTERS ==========")

print(clusters)


# =====================================
# 7. DISPLAY CLUSTER CENTERS
# =====================================

print("\n========== CLUSTER CENTERS ==========")

print(model.cluster_centers_)


# =====================================
# 8. VISUALIZE
# =====================================

plt.figure()

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.title("K-Means Clustering")

plt.show()