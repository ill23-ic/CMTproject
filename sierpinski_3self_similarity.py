import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


def build_sierpinski(G):
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

