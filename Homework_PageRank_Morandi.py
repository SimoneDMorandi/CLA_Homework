"""
Computational Linear Algebra For Large Scale Problems.
A.Y. 2025-2026.
Homework: PageRank. Main file.
Created by Simone Domenico Morandi (336974).

The following code, along with the custom functions, implements two versions of the PageRank algorithm. The first one
is a simplified version that do not account for the damping factor "m", while the second one is the complete version of
the algorithm. Both algorithms are applied to several simple webs of pages and to a dataset of 6012 pages in order to
simulate a more realistic scenario in which PageRank algorithm can be used. The dataset presents dangling nodes that are
carefully handled in order for the algorithm to converge. We also analyze how the presence of "m" affects the importance
scores computed by the algorithm and, in addiction, how different values of the damping factor affect computation times
and final scores.
"""
########################################################################################################################


# Basic imports.
from PageRank_Functions import * # Importing custom functions.
import numpy as np
import time
import matplotlib.pyplot as plt

# %% PageRank algorithm - Simplified version.

"""
In this section we apply a simplified version of PageRank algorithm to Web 1, Web 2 and a dataset.
For Web 1 and Web 2, importance scores are shown, while for the dataset only relevant results are shown.
"""

# Web 1.

A_1 = np.array([
           [0, 0, 1, 1],
           [1, 0, 0, 0],
           [1, 1, 0, 1],
           [1, 1, 0, 0]], dtype=float)
# Computing scores.
r_1 = pagerank(A_1,0)
# Displaying scores.
print("PageRank - Simplified version:\n")
print("Displaying scores of Web 1: \n")
for i, score in enumerate(r_1):
    print(f"Page {i + 1}: {score:6f}")
print("\n")

# Web 2.

A_2 = np.array([
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0]], dtype=float)
# Computing scores.
r_2 = pagerank(A_2,0)
# Displaying scores.
print("Displaying scores of Web 2: \n")
for i, score in enumerate(r_2):
    print(f"Page {i+1}: {score:6f}")
print("\n")

# Dataset.

A_dat = build_A("hollins.dat")
# Computing scores.
r_3 = pagerank(A_dat,0)
# Counting components that are zero.
count_zeros = np.count_nonzero(r_3 < 1e-10)
print("Displaying number of components that are zero: \n")
print(count_zeros)
print("\n")
# Retrieving most important page.
max_index = np.argmax(r_3)
max_3 = np.max(r_3)
print("Most important page with score: \n")
print(f"Page: {max_index+1}, with {max_3:6f} score.")
print("\n")


# %% PageRank algorithm - Full version.

"""
In this section we apply the PageRank algorithm (with damping factor m = 0.15) to Web 1, Web 2 and a dataset.
For Web 1 and Web 2 importance scores are shown, while for dataset only relevant results are shown.
"""

# Web 1.

# Computing scores.
r_1_tot = pagerank(A_1,0.15)
# Displaying scores.
print("Displaying scores of Web 1: \n")
for i, score in enumerate(r_1_tot):
    print(f"Page {i+1}: {score:.6f}")
print("\n")

# Web 2.

# Computing scores.
r_2_tot = pagerank(A_2,m=0.15)
# Displaying scores.
print("Displaying scores of Web 2: \n")
for i, score in enumerate(r_2_tot):
    print(f"Page {i+1}: {score:6f}")
print("\n")

# Dataset.

# Computing scores.
r_3_tot = pagerank(A_dat, m=0.15)
# Counting components that are zero.
count_zeros_1 = np.count_nonzero(r_3_tot < 1e-10)
print("Displaying number of components that are zero: \n")
print(count_zeros_1)
print("\n")
# Retrieving most important page.
max_index_1 = np.argmax(r_3_tot)
max_3_1 = np.max(r_3_tot)
print("Most important page with score: \n")
print(f"Page: {max_index_1+1}, with {max_3_1:6f} score.")
print("\n")

# Comparison between scores on dataset with m = 0 and m = 0.15

# Sorted scores distribution.
plt.figure(figsize=(10,6))
# Sorting scores.
r_3_sorted = np.sort(r_3)[::-1].copy() # m=0.
r_3_tot_sorted = np.sort(r_3_tot)[::-1].copy() # m=0.15.

plt.plot(r_3_sorted, label='m=0')
plt.plot(r_3_tot_sorted, label='m=0.15')
plt.yscale('log')  # Log-scale for better visualization.
plt.xlabel('Pages (sorted by rank)')
plt.ylabel('Scores (log scale)')
plt.title('Scores Distributions.')
plt.legend()
plt.grid(True)
plt.show()


# %% Exercise 11-12.

"""
In this section we solve exercises 11 and 12 from the proposed article on PageRank algorithm.
Both consist in modifying Web 1 and computing importance scores with the function defined early for
the PageRank algorithm with both m = 0 and m = 0.15.
"""

# Exercise 11.
A_4 = np.array([
           [0, 0, 1, 1, 0],
           [1, 0, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 0, 0],
           [0, 0, 1, 0, 0]], dtype=float)
# Computing scores.
r_4_tot = pagerank(A_4,m=0.15)
# Displaying scores.
print("Displaying scores of Web 1-a: \n")
for i, score in enumerate(r_4_tot):
    print(f"Page {i+1}: {score:6f}")
print("\n")

# Exercise 12.
A_5 = np.array([
           [0, 0, 1, 1, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 1, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 0, 0, 0, 0]], dtype=float)
# Computing scores.
r_5 = pagerank(A_5,0)
r_5_tot = pagerank(A_5,m=0.15)
# Displaying scores.
print("Displaying scores of augmented Web 1-b: \n")
print("PageRank, Simplified Version.")
for i, score in enumerate(r_5):
    print(f"Page {i+1}: {score:6f}")
print("\n")
print("PageRank:")
for i, score in enumerate(r_5_tot):
    print(f"Page {i+1}: {score:6f}")
print("\n")


# %% Exercise 17.

"""
In this section we solve exercises 17 from the proposed article on PageRank algorithm.
The exercise consists in taking different values of "m" and see how importance scores and computation times change.
We plot the different times of execution with respect to "m" and we track the ranking of 10 random pages in order to 
highlight how different values of "m" affect the final scores. Finally, we show how the variance of the score vector 
changes as m increases.
"""

# Choosing different values of m.
m_values = np.array([0.01,0.1,0.3,0.5,0.7,0.85,0.9,0.99], dtype=float)
# Initializing collectors.
results_dat = []
times_dat = []
var_dat =  []
for m in m_values:
    # Computing score results and saving time of execution and variance.
    start = time.time()
    r = pagerank(A_dat,m)
    end = time.time()
    results_dat.append(r)
    times_dat.append(end-start)
    var_dat.append(np.var(r))

# Plotting computation times.

plt.figure(figsize=(8,5))
plt.plot(m_values, times_dat, 'o-', color='teal')
plt.xlabel('Damping factor (m)')
plt.ylabel('Computation time (s)')
plt.title('Computation Time vs Damping Factor')
plt.grid(True)
plt.show()

# Tracking top pages.

# Choosing reference damping factor and top-k pages.
ref_m = m_values[0]
ref_index = np.where(np.isclose(m_values, ref_m))[0][0]
ref_rank = results_dat[ref_index]
top_k = 10  # Number of top pages to track.
# Finding indices of the top-k pages at reference m.
num_pages = len(ref_rank)
tracked_pages = np.random.choice(np.arange(num_pages), size = top_k, replace = False)
print(f"Tracking {top_k} pages from m = {ref_m}")
print("Tracked pages:", tracked_pages)

# Computing ranking positions of those pages for all m-values.
top_ranks = np.zeros((len(m_values), top_k))
for i, r in enumerate(results_dat):
    order = np.argsort(r)[::-1]  # Descending order of scores.
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)  # Ranking positions (1 = best).
    top_ranks[i, :] = ranks[tracked_pages]


# Plotting ranking position changes.
plt.figure(figsize=(10,6))
for j in range(top_k):
    plt.plot(m_values, top_ranks[:, j], marker='o', label=f'Page {tracked_pages[j]}')
plt.xlabel('Damping factor (m)')
plt.ylabel('Rank position (lower = better)')
plt.title(f'Ranking Positions of Top {top_k} Pages vs Damping Factor.')
plt.gca().invert_yaxis() # Invert order.
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plotting variance.
plt.figure(figsize=(8,5))
plt.plot(m_values, var_dat, 'o-', color='teal')
plt.xlabel('Damping factor (m)')
plt.ylabel('Variance of scores')
plt.title('Variance of scores vs Damping Factor.')
plt.grid(True)
plt.show()



########################################################################################################################