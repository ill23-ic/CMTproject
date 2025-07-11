import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def build_sierpinski(G):
    # Level-0: triangle of 3 nodes
    nodes = [(0,0), (1,0), (0.5, np.sqrt(3)/2)]
    edges = [(0,1), (1,2), (2,0)]
    for g in range(1, G):
        N = len(nodes)
        coords = np.array(nodes)
        # Scale and shift previous triangle into 3 corners
        new_edges = []
        for offset in [(0,0), (0.5,0), (0.25, np.sqrt(3)/4)]:
            #NumPy automatically expands offset to match coords’s shape
            new_coords = (coords + offset) / 2
            nodes.extend([tuple(p) for p in new_coords])
            # Only replicate edges from the previous generation
            off = len(nodes) - N
            new_edges.extend([(u+off, v+off) for u,v in edges[-3*N//3:]])
        edges.extend(new_edges)
    return nodes, edges

def tight_binding_sierpinski(G, t=1.0):
    nodes, edges = build_sierpinski(G)
    N = len(nodes)
    rows, cols, vals = [], [], []
    for i,j in edges:
        rows += [i, j]; cols += [j, i]; vals += [-t, -t]
    H = sp.coo_matrix((vals, (rows, cols)), shape=(N,N)).tocsr()
    return H, nodes

def plot_tb_sierpinski(G, t=1.0, k=6):
    H, nodes = tight_binding_sierpinski(G, t)
    eigvals, eigvecs = eigsh(H, k=k, which='SA')
    nodes = np.array(nodes)
    figs = []
    for i in range(k):
        psi = eigvecs[:, i]
        fig, ax = plt.subplots(figsize=(6,6))
        sc = ax.scatter(nodes[:,0], nodes[:,1], c=np.abs(psi)**2,
                        cmap='viridis', s=100)
        ax.set_title(f"Gen {G} State {i}, E={eigvals[i]:.3f}")
        plt.colorbar(sc, ax=ax, label='Probability')
        figs.append(fig)
    plt.show()
    return eigvals, figs

# Calculate and plot states for generation-3 Sierpinski triangle
energies, figures = plot_tb_sierpinski(G=3, t=1.0, k=6)