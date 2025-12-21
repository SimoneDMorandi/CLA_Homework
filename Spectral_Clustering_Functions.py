"""
Computational Linear Algebra For Large Scale Problems.
A.Y. 2025-2026.
Homework: PageRank. Functions file.
Created by Simone Domenico Morandi (336974).

The following code three functions are defined.
The first one creates a dataset made out of separated vertical bars.
The second one applies the spectral clustering technique with different modalities to a dataset.
The third one computes the laplacian graph of an n-dimensional hypercube.
"""
########################################################################################################################


# Basic imports.
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances # For distance matrix.


def make_bars(n,x_pos,y_range,x_noise,random_state):
    """
    Generates a dataset of vertical bar-shaped clusters.
    Parameters:
    n: number of samples,
    x_pos: centres of the bars,
    y_range : vertical range of the bars,
    x_noise : data noise,
    random_state : seed.
    Returns:
    X : final dataset,
    Y : labels of the final dataset.
    The function, after setting the random seed, divides the number of total points into equal parts, then
    generates vertical bars and store them into the final dataset.
    """

    # Setting seed.
    if random_state is not None:
        np.random.seed(random_state)

    k = len(x_pos) # Number of bars
    base = n // k
    X_list = []
    Y_list = []
    for label, x0 in enumerate(x_pos):
        # Number of points in each bar.
        N = base if label < k-1 else (n - base*(k-1))
        # Generating points.
        xs = x0 + x_noise * np.random.randn(N)
        ys = np.random.uniform(y_range[0], y_range[1], N)
        # Building dataset.
        X_list.append(np.column_stack([xs, ys]))
        Y_list.append(np.full(N, label))
    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)
    return X,Y


def spectral_clustering(X, k, n_neighbors, sigma, Norm, random_state):
    """
    Applies spectral clustering to a dataset.
    Parameters:
    X : input data,
    k : number of clusters,
    n_neighbors : number of nearest neighbors to use for the neighborhood graph,
    sigma : distance,
    Norm : boolean to state whether one wants the normalized laplacian graph.
    random_state: seed.
    Returns:
    labels : cluster labels from Kmeans,
    eigval : the sorted eigenvalues.
    eigvec : the first smallest "k" eigenvectors.
    First, the function builds the weight matrix for both neighborhood graph or fully connected graph according to
    the value of "n_neighbors". Then, builds the graph Laplacian or the normalized graph Laplacian depending on "Norm"
    value. Lastly, computes eigenvalues of the graph Laplacian and applies Kmeans to the first smallest "k" eigenvectors
    of the graph Laplacian, using an auxiliary matrix "U".
    """

    X = np.asarray(X)
    n_samples = X.shape[0]

    # Building weight matrix.
    if n_neighbors is not None and n_neighbors < n_samples:
        # Neighborhood graph.
        A = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False).toarray()
        A[A==0] = np.inf
        W = np.exp(-A ** 2 / (2 * sigma ** 2))
        W[A==np.inf] = 0
        W = np.maximum(W, W.T)  # Symmetrizing.
    else:
        # Fully connect graph.
        dist = pairwise_distances(X)
        sigma = np.median(dist[dist > 0])
        W = np.exp(-dist ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        W = np.maximum(W, W.T) # Symmetrizing.

    # Computing graph Laplacian.
    degrees = W.sum(axis=1)

    if Norm:
        # Normalized symmetric graph Laplacian.
        degrees[degrees == 0] = 1e-10
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees, where=degrees>0, out=None))
        L = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        # Standard graph Laplacian.
        D = np.diag(degrees)
        L = D - W

    # Computing eigenvalues and eigenvectors.
    eigval_all, eigvec_all = np.linalg.eigh(L)
    idx = np.argsort(eigval_all)
    eigvec = eigvec_all[:, idx[:k]] # Taking the first "k" eigenvectors.

    # Storing eigenvectors according to what graph Laplacian one has chosen.
    if Norm:
        U = eigvec / np.linalg.norm(eigvec, axis=1, keepdims=True)
    else:
        U = eigvec
    U[np.isnan(U)] = 0 # For safety.

    # Applying Kmeans to the eigenvectors.
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(U)

    return labels, eigval_all, U


def laplacian_hypercube(n):
    """
    Computes the graph Laplacian associated to an n-hypercube.
    Parameters:
    n : Dimension of the hypercube.
    Returns:
    L : The graph Laplacian associated to the n-hypercube.
    The function recursively calls itself until n==1, then the first graph Laplacian for the 1-dimensional hypercube is
    built. Then builds the final graph Laplacian by using a block matrix.
    """
    # Base block.
    if n == 1:
        return np.array([[1, -1],
                         [-1, 1]], dtype=float)
    L_prev = laplacian_hypercube(n-1) # Recursive call.
    # Assembling final Laplacian.
    I = np.eye(2**(n-1))
    L =  np.block([
        [L_prev + I, -I],
        [-I, L_prev + I]
    ])
    return L



########################################################################################################################