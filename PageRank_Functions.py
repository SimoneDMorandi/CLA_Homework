"""
Computational Linear Algebra For Large Scale Problems.
A.Y. 2025-2026.
Homework: PageRank. Functions file.
Created by Simone Domenico Morandi (336974).

In the following code two functions are defined.
The first one implements the PageRank algorithm for both dense and sparse matrices, while accounting for dangling nodes.
The second function computes the adjacency matrix from a ".dat" file written in a specific format.
"""
########################################################################################################################


# Basic imports.
import numpy as np
import scipy.sparse as sps
from scipy import sparse

def pagerank(A, m, tol=1e-12, max_iter=1000):
    """
    Computes the PageRank vector accounting for the damping factor and dangling nodes.
    Parameters:
    A : Adjacency matrix,
    m : damping factor,
    tol : tolerance,
    max_iter : maximum number of iterations.
    Returns:
    r : PageRank vector.
    After initializing "r" and making "A" a columns stochastic matrix while accounting for dangling nodes if the matrix
    is sparse, we apply the power method in order to compute the eigenvector associated with the greatest eigenvalue;
    for the full version, the iteration matrix is defined as a convex combination of "A" and "S", an auxiliary vector,
    with "m" as parameter. A check for convergence is done by confronting the norm of the eigenvectors computed at the
    new and the previous iteration. In addiction to the maximum number of iteration, this check represents a stopping
    criterion. The i-th component of "r" is the importance score of the i-th page.
    """

    n = A.shape[0]
    if sparse.issparse(A):
        A = A.tocsc()
        col_sums = np.asarray(A.sum(axis=0)).ravel()
        # Handling dangling nodes.
        dangling = (col_sums == 0)
        col_sums[col_sums == 0] = 1
        A = A.multiply(1.0 / col_sums) # To preserve sparsity.
    else:
        col_sums = A.sum(axis=0)
        dangling = (col_sums == 0) # Needed for m != 0.
        col_sums[col_sums == 0] = 1
        A = A / col_sums

    # Power method.
    r = np.ones(n) / n
    s = np.ones(n) / n
    for i in range(max_iter):
        if m == 0:
            r_new = A @ r # Applying simple PageRank iteration.
        else:
            dangling_mass = r[dangling].sum() # Computing dangling nodes' contribution.
            r_new =  (1 - m) * (A @ r) +m*s +(1-m)*dangling_mass*s # Applying PageRank iteration.
        r_new /= np.linalg.norm(r_new,1) # Normalizing.
        # Deciding whether to stop according to tolerance.
        if np.linalg.norm(r_new-r,1) < tol:
            break
        r = r_new

    # Checking residual (convergence for eigenvalue).
    """
    dangling_mass = r[dangling].sum()
    residual = np.linalg.norm( (1 - m) * (A @ r + dangling_mass * s) + m * s - r,1)
    print("Eigenvalue residual:", residual)
    """
    return r.flatten()


def build_A(filename):
    """
    Constructs adjacency matrix from a ".dat" file.
    The .dat file format:
        number_of_pages, number_of_links
        id url
        from (page) to (page)
    Parameters:
        filename: the name of the .dat file.
    Returns:
        A : adjacency matrix.
    In the first line of the file we find the number of pages and link. Then a list of all pages is read, which we can
    ignore since it is not helpful to construct "A". Then we have a list of all links with "from-to" format.
    After reading thw whole file and storing information, we can start building our adjacency matrix.
    We initialize the matrix as sparse since the number of links is much lower than the dimension of the matrix.
    """

    with open(filename, 'r') as f:
        # Reading first line: number of pages and number of links.
        first_line = f.readline().strip().split()
        n_pages, n_links = map(int, first_line)

        # Skipping the list of URLs.
        for i in range(n_pages):
            f.readline()

        # Reading all "from to" lines.
        from_nodes = []
        to_nodes = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                from_nodes.append(int(parts[0]))
                to_nodes.append(int(parts[1]))

    # Initializing A.
    A = sps.lil_matrix((n_pages, n_pages), dtype=int) # Using LIL storage scheme since A is sparse.

    # Filling entries.
    for f_node, t_node in zip(from_nodes, to_nodes):
        A[t_node - 1, f_node - 1] = 1 # A_{i,j} = 1 if from (i) to (j) exists.

    return A


########################################################################################################################