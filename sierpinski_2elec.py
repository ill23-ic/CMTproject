import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_sierpinski_2(G):
    # Level-0: Base triangle
    nodes = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
    edges = [(0, 1), (1, 2), (2, 0)]
    
    for g in range(1, G):
        N_prev = len(nodes)
        new_nodes = []
        new_edges = []
        
        # Generate 3 scaled/offset copies of the current structure
        for i, offset in enumerate([(0, 0), (0.5, 0), (0.25, np.sqrt(3)/4)]):
            # Scale and shift nodes
            shifted_nodes = [(x/2 + offset[0], y/2 + offset[1]) 
                           for (x, y) in nodes[:N_prev]]
            new_nodes.extend(shifted_nodes)
            
            # Reindex edges with offset
            edge_offset = i * N_prev
            new_edges.extend([(u + edge_offset, v + edge_offset) 
                            for (u, v) in edges[:N_prev]])
        
        # Merge duplicate nodes (round coordinates to avoid floating-point errors)
        unique_nodes = {}
        node_mapping = {}  # old index → new index
        for idx, (x, y) in enumerate(new_nodes):
            # Round to 10 decimal places to handle floating-point precision
            key = (round(x, 10), round(y, 10))
            if key not in unique_nodes:
                unique_nodes[key] = (x, y)
            node_mapping[idx] = len(unique_nodes) - 1  # New index
        
        # Reindex edges to use merged nodes
        merged_edges = set()
        for u, v in new_edges:
            new_u, new_v = node_mapping[u], node_mapping[v]
            if new_u != new_v:  # Avoid self-loops
                merged_edges.add((min(new_u, new_v), max(new_u, new_v)))
        
        # Update nodes and edges
        nodes = list(unique_nodes.values())
        edges = list(merged_edges)
    
    return nodes, edges

def tight_binding_sierpinski_2(G, t=1.0):
    """
    Code basically performs

    H[rows[i],cols[i]]=vals[i] for i in range (len(vals))

    implicitly
    """
    nodes, edges = build_sierpinski_2(G)  # Uses the corrected fractal generator
    N = len(nodes)
    rows, cols, vals = [], [], []
    for i, j in edges:
        rows += [i, j]; cols += [j, i]; vals += [-t, -t]  # Hopping terms
    H = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return H, nodes, edges

def plot_tb_sierpinski_2(G, t=1.0, k=6):
    H, nodes, edges = tight_binding_sierpinski_2(G, t)
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