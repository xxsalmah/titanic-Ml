from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# =====================================
# 1. LOAD DATA
# =====================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================
# 2. SCALE DATA
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =====================================
# 3. CREATE PCA
# =====================================

pca = PCA(n_components=2)


# =====================================
# 4. TRANSFORM DATA
# =====================================

X_pca = pca.fit_transform(X_scaled)


# =====================================
# 5. DISPLAY SHAPE
# =====================================

print("========== ORIGINAL DATA ==========")

print("Shape:", X.shape)


print("\n========== PCA DATA ==========")

print("Shape:", X_pca.shape)


# =====================================
# 6. EXPLAINED VARIANCE
# =====================================

print("\n========== EXPLAINED VARIANCE ==========")

print(
    pca.explained_variance_ratio_
)

print(
    "Total:",
    round(
        pca.explained_variance_ratio_.sum(),
        3
    )
)


# =====================================
# 7. VISUALIZE
# =====================================

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Iris Dataset After PCA")

plt.colorbar()

plt.show()