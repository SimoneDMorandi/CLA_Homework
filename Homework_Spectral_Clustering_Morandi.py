"""
Computational Linear Algebra For Large Scale Problems.
A.Y. 2025-2026.
Homework: Spectral Clustering. Main file.
Created by Simone Domenico Morandi (336974).

The following code, along with the custom functions, implements the Kmeans algorithm to some datasets. Then, applies the
spectral clustering technique to the same datasets. After that, we check some theoretical results and solve two
exercises regarding the graph Laplacian of the n-cycle and the n-hypercube.
"""
########################################################################################################################


# Basic imports.
from Spectral_Clustering_Functions import * # Importing custom functions.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score # Similarity between clusterings.

# %% Limitations of Kmeans.

"""
In this section we show the limitations of the Kmeans algorithm by applying it to different datasets and showing that 
the clusterization is not optimal. Then, we apply the spectral clustering procedure to the same datasets and we obtain a 
better clusterization. 
"""

# Generating dataset.
n = 1000
random_state = 42

# Generating Gaussian Mixture Model (GMM).
X_GMM, Y_GMM = make_blobs(n_samples=n, centers=3, cluster_std=0.9, random_state=random_state)
# Generating concentric circles data.
X_circ,Y_circ = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=random_state)
# Generating anisotropic data (vertical bars).
x_pos = (-6,0,6) # Center of the bars.
y_range = (-10,15)
x_noise = 0.3
X_an,Y_an = make_bars(n,x_pos,y_range,x_noise,random_state) # Genereting bars.
dataset = [
    (X_GMM, Y_GMM, "Gaussian Mixture Model"),
    (X_circ, Y_circ, "Concentric Circles"),
    (X_an, Y_an, "Anisotropic Data"),
]

# Plotting results.
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# Applying K-means to dataset.
for i, (X, Y_true, name) in enumerate(dataset):
    k = 2 if name == "Concentric Circles" else 3
    kmeans = KMeans(k, n_init=10, random_state=random_state)
    # Plotting original dataset.
    axes[i, 0].scatter(X[:, 0], X[:, 1], color = "blue", s=20,)
    axes[i, 0].set_title(f"Original dataset: {name}.")
    axes[i,0].set_xticklabels([])
    axes[i, 0].set_yticklabels([])
    # Plotting clustering.
    Y_predict = kmeans.fit_predict(X)
    score = adjusted_rand_score(Y_true, Y_predict)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=Y_predict, s=20, cmap="viridis")
    axes[i, 1].set_title(f"K-means.")
    axes[i, 1].set_xticklabels([])
    axes[i, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

# %% Spectral clustering algorithm.

# Plotting results.
total_eigval = []
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for i, (X, Y_true, name) in enumerate(dataset):
    k = 2 if name == "Concentric Circles" else 3
    n_neighbors = [None,None,None]
    sigma = [1.0,0.2,1.5]
    labels, eigval, U = spectral_clustering(X, k,
                                            n_neighbors=n_neighbors[i], # If None, fully connected graph.
                                            sigma=sigma[i],
                                            Norm=False,
                                            random_state = random_state)  # Normalized (True) or un-normalized (False) graph Laplacian.
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap="viridis")
    axes[i].set_title(f"Spectral clustering - {name}")
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
    total_eigval.append(eigval)
plt.tight_layout()
plt.show()

# Printing eigenvalues of L.
fig,axes = plt.subplots(3, 1, figsize=(10, 12))
for i,eigval in enumerate(total_eigval):
    axes[i].plot(np.arange(len(eigval)),eigval, '-o')
    axes[i].set_ylabel("Eigenvalue")
plt.tight_layout()
plt.show()


# %% Exercises on graph Laplacian.

"""
In the first part of this section, we compute different adjacency matrix and see how the eigenvalues and eigenvectors
of the graph Laplacian change. Then in the second part we solve two different problems. The first one is to verify that 
the discrete sine and cosine functions are eigenvectors of the graph Laplacian associated to the n-cycle.
The second one consist in finding the spectrum of the graph Laplacian associated to the n-hypercube.
"""

# Fixing dimension of the matrices for all exercises.
n = 5

# Exercise 0-a.

# Defining matrices.
A_1 = np.ones((n, n)) -np.eye(n)
D_1 = (n-1)*np.eye(n)
# Computing L, its eigenvalues and eigenvectors.
L_1 = D_1-A_1
eigval_1, eigvec_1 = np.linalg.eigh(L_1)

# Verifying results.

# Eigenvalues.
print("Eigenvalues:", np.round(eigval_1, 6))
# Orthogonality.
check_eigval = np.round(eigvec_1.T @eigvec_1)
print("Checking orthogonality:",np.linalg.norm(np.eye(n)-check_eigval))
# Eigenvectors.
check_norm_1 = np.linalg.norm(eigvec_1[:,0])
print("First eigenvector (for λ=0):", np.round(eigvec_1[:,0]/check_norm_1, 6))
print("Sum of entries in first eigenvector:", np.round(np.sum(eigvec_1[:,0]),6))
print(f"Square root of {n}:", np.round(np.sqrt(n),6))
print("\n")

# Exercise 0-2.

# Defining matrices.
diag_2 = np.ones(n-1)
A_2 = np.zeros((n,n)) + np.diag(diag_2,1) + np.diag(diag_2,-1)
D_2 = 2*np.eye(n)
D_2[0,0] = 1
D_2[-1,-1] = 1
# Computing L, its eigenvalues and eigenvectors.
L_2 = D_2-A_2
eigval_2, eigvec_2 = np.linalg.eigh(L_2)
lambda_k_2 = [2 - 2*np.cos((np.pi*k)/n) for k in range(n)]
# Verifying results.
print("Numerical eigenvalues:", np.round(eigval_2, 6))
print("Theoretical eigenvalues:", np.round(lambda_k_2, 6))
print("\n")

# Exercise 1 - n-cycle.

# Defining matrices.
A_3 = A_2.copy()
A_3[n-1,0] = 1
A_3[0,n-1] = 1
D_3 = 2*np.eye(n)
# Computing L, its eigenvalues and eigenvectors.
L_3 = D_3-A_3
eigval_3, eigvec_3 = np.linalg.eigh(L_3)
lambda_k_3 = [2-2*np.cos((2*np.pi*k)/n) for k in range(n)]

# Verifying results.

# Eigenvalues.
print("Numerical eigenvalues:", np.round(eigval_3, 6))
print("Theoretical eigenvalues:", np.round(lambda_k_3, 6))
print("\n")
# Eigenvectors.
k = 4
lambda_k = 2-2*np.cos(2*np.pi*k/n)
phi_k = np.cos(2*np.pi*k*np.arange(n)/n) # For k=2.
v_k = eigvec_3[:,k]
# Normalizing.
phi_k /= np.linalg.norm(phi_k)
v_k /= np.linalg.norm(v_k)
# Computing similarity.
residual = np.linalg.norm(L_3 @ phi_k - lambda_k * phi_k)
print(f"Residual for k={k}:", np.round(residual,6))
print("\n")

# Verifying discrete sine and cosine.

# Defining discrete sine and cosine.
j = np.arange(n)
xi_k = np.cos(2*np.pi*k*j/n)
psi_k = np.sin(2*np.pi*k*j/n)
# Applying Laplacian.
L_xi = L_3 @ xi_k
L_psi = L_3 @ psi_k
# Comparing.
error_phi = np.linalg.norm(L_xi - lambda_k*xi_k)
error_psi = np.linalg.norm(L_psi - lambda_k*psi_k)
print(f"||Lξ - λξ|| =", np.round(error_phi,6))
print(f"||Lψ - λψ|| =", np.round(error_psi,6))
print("\n")
# Plotting results.
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
# Plotting discrete sine.
axes[0].plot(j, psi_k, marker='o', label="psi_k (sine eigenvector)")
axes[0].plot(j, L_psi/lambda_k, '--', marker='x', label="(L psi_k) / λ")
axes[0].set_title("Sine eigenvector vs (L psi_k) / λ.")
axes[0].set_xlabel("j")
axes[0].grid(True)
axes[0].legend()
# Plotting discrete cosine.
axes[1].plot(j, xi_k, marker='o', label="xi_k (cosine eigenvector)")
axes[1].plot(j, L_xi/lambda_k, '--', marker='x', label="(L xi_k) / λ")
axes[1].set_title("Cosine eigenvector vs (L xi_k) / λ.")
axes[1].set_xlabel("j")
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.show()

# Exercise 2 - n-hypercube.

# Computing graph Laplacian.
L_4 = laplacian_hypercube(n)
# Computing eigenvalues.
eigval_4, eigvec_4 = np.linalg.eigh(L_4)
# Printing unique eigenvalues.
print(f"Eigenvalues of the {n}-dimensional hypercube Laplacian:", np.unique(np.round(eigval_4,6)))
print("\n")
# Plotting results.
k = np.arange(1, len(eigval_4)+1)
plt.figure(figsize=(8,5))
plt.scatter(k, eigval_4, color='blue')
plt.xlabel('Index (k)')
plt.ylabel('Eigenvalue λ_k')
plt.title(f'Eigenvalues of {n}-dimensional hypercube Laplacian.')
plt.show()


########################################################################################################################