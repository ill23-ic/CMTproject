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
        
        # Generate 3 scaled/offset copies
        for i, offset in enumerate([(0, 0), (0.5, 0), (0.25, np.sqrt(3)/4)]):
            # Scale and shift nodes
            shifted_nodes = [(x/2 + offset[0], y/2 + offset[1]) 
                           for (x, y) in nodes]
            new_nodes.extend(shifted_nodes)
            
            # Reindex edges with offset
            edge_offset = i * N_prev
            new_edges.extend([(u + edge_offset, v + edge_offset) 
                            for (u, v) in edges])
        
        # Deduplicate nodes and build mapping
        coord_to_index = {}
        dedup_nodes = []
        old_to_new = {}  # Old index → new index
        
        for old_idx, (x, y) in enumerate(new_nodes):
            key = (round(x, 10), round(y, 10))
            if key not in coord_to_index:
                coord_to_index[key] = len(dedup_nodes)
                dedup_nodes.append((x, y))
            old_to_new[old_idx] = coord_to_index[key]
        
        # Rebuild edges using new indices
        dedup_edges = set()
        for u, v in new_edges:
            new_u, new_v = old_to_new[u], old_to_new[v]
            if new_u != new_v:  # Remove self-loops
                dedup_edges.add(tuple(sorted((new_u, new_v))))  # Ensure unique representation
        
        # Verify edges are nearest-neighbors
        final_edges = []
        node_array = np.array(dedup_nodes)
        for u, v in dedup_edges:
            dist = np.linalg.norm(node_array[u] - node_array[v])
            expected_dist = 1.0/(2**g)  # Scaled nearest-neighbor distance
            if np.isclose(dist, expected_dist, atol=1e-5):
                final_edges.append((u, v))
        
        nodes = dedup_nodes
        edges = final_edges
    
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

eigvals, figs = plot_tb_sierpinski_2(G=2, k=1)