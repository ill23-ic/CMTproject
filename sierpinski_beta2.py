import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_sierpinski(G):
    """
    Self similarity in all 3 base nodes, G is number of generations, base is 1, not 0.
    Interactions occur within nearest neighbour within the same zoomed in triangle. Inaccurate for low G.
    """
    # Level-0: Base triangle (3 nodes)
    nodes = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
    edges = [(0, 1), (1, 2), (2, 0)]
    
    for g in range(1, G):
        N_prev = len(nodes)  # Nodes before this generation
        new_nodes = []
        new_edges = []
        
        # Three copies of the previous generation
        for i in range(3):
            # Calculate offset for this copy
            if i == 0:
                offset = (0, 0)  # Bottom-left
            elif i == 1:
                offset = (0.5, 0)  # Bottom-right
            else:
                offset = (0.25, np.sqrt(3)/4)  # Top-center
            
            # Scale and shift previous nodes
            shifted_nodes = [(x/2 + offset[0], y/2 + offset[1]) 
                           for (x, y) in nodes[:N_prev]]
            new_nodes.extend(shifted_nodes)
            
            # Replicate edges with index offset
            edge_offset = i * N_prev
            new_edges.extend([(u + edge_offset, v + edge_offset) 
                             for (u, v) in edges[:N_prev]])
        
        nodes = new_nodes  # Update nodes
        edges = new_edges  # Update edges
    
    return nodes, edges

def tight_binding_sierpinski(G, t=1.0):
    """
    Code basically performs

    H[rows[i],cols[i]]=vals[i] for i in range (len(vals))

    implicitly
    """
    nodes, edges = build_sierpinski(G)  # Uses the corrected fractal generator
    N = len(nodes)
    rows, cols, vals = [], [], []
    for i, j in edges:
        rows += [i, j]; cols += [j, i]; vals += [-t, -t]  # Hopping terms
    H = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return H, nodes, edges

def plot_tb_sierpinski(G, t=1.0, k=6):
    H, nodes, edges = tight_binding_sierpinski(G, t)
    eigvals, eigvecs = eigsh(H, k=k, which='SA')  # Smallest-algebraic eigenvalues
    nodes = np.array(nodes)  # Convert to NumPy array
    
    # Plot eigenstates
    figs = []
    for i in range(k):
        psi = eigvecs[:, i]  # Eigenstate i
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=np.abs(psi)**2, 
                        cmap='viridis', s=100)
        ax.set_title(f"Gen {G}, State {i}, E={eigvals[i]:.3f}")
        plt.colorbar(sc, ax=ax, label='Probability')
        figs.append(fig)
    plt.show()
    return eigvals, figs

def plot_tb_sierpinski_3d(G, t=1.0, k=6):
    H, nodes, edges = tight_binding_sierpinski(G, t)
    eigvals, eigvecs = eigsh(H, k=k, which='SA')
    nodes = np.array(nodes)  # Shape: (N, 2)
    
    figs = []
    for i in range(k):
        psi = eigvecs[:, i]
        prob = np.abs(psi)**2
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize probabilities for color mapping
        prob_normalized = prob / prob.max()
        colors = plt.cm.viridis(prob_normalized)  # Color bars by probability
        
        # Plot 3D bars
        dx = dy = 0.02  # Width/depth of bars (adjust based on node spacing)
        ax.bar3d(
            nodes[:, 0],          # x-coordinates
            nodes[:, 1],          # y-coordinates
            np.zeros_like(prob),  # Base of bars (z=0)
            dx, dy,               # Bar width/depth
            prob,                 # Bar height (z)
            color=colors,
            edgecolor='k',
            alpha=0.8
        )
        
        # Add edges (optional)
        for (u, v) in edges:
            ax.plot(
                [nodes[u, 0], nodes[v, 0]],
                [nodes[u, 1], nodes[v, 1]],
                [0, 0],  # Edges at base (z=0)
                color='gray', linewidth=0.5, alpha=0.5
            )
        
        ax.set_title(f"Gen {G}, State {i}, E={eigvals[i]:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('$|\psi|^2$')
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, prob.max()))
        mappable.set_array(prob)
        fig.colorbar(mappable, ax=ax, label='$|\psi|^2$')
        
        figs.append(fig)
    
    plt.tight_layout()
    plt.show()
    return eigvals, figs


# Implementation

eigvals, figs = plot_tb_sierpinski_3d(G=3, k=1)

#eigvals, figs = plot_tb_sierpinski(G=5, k=3)