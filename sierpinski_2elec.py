import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sierpinski_3self_similarity import tight_binding_sierpinski

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
        dx = dy = 0.01  # Width/depth of bars (adjust based on node spacing)
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

eigvals, figs = plot_tb_sierpinski_3d(G=5, k=1)