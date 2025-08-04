#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:58:38 2025

@author: tinlokisaaclai
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


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
        rows += [i, j]
        cols += [j, i]
        vals += [-t, -t]  # Hopping terms
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

def build_two_particle_H(H1, U):
    """
    Builds the full two-particle Hamiltonian with on-site interaction U.
    """
    N = H1.shape[0]
    I = sp.identity(N, format='csr')
    
    # Kinetic part: H1 ⊗ I + I ⊗ H1
    H_kin = sp.kron(H1, I) + sp.kron(I, H1)
    
    # Interaction: U * sum_i |i,i><i,i|
    row = []
    col = []
    data = []
    for i in range(N):
        idx = i * N + i  # index of |i, i>
        row.append(idx)
        col.append(idx)
        data.append(U)
    H_int = sp.coo_matrix((data, (row, col)), shape=(N * N, N * N)).tocsr()
    
    H_total = H_kin + H_int
    return H_total


def plot_two_particle_state(psi, nodes, title=''):
    N = len(nodes)
    psi_reshaped = psi.reshape((N, N))  # psi[i, j] = amplitude at |i, j>
    prob = np.abs(psi_reshaped)**2

    node_array = np.array(nodes)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    for i in range(N):
        for j in range(N):
            x = (node_array[i, 0] + node_array[j, 0]) / 2
            y = (node_array[i, 1] + node_array[j, 1]) / 2
            size = 300 * prob[i, j] / prob.max()
            if size > 1e-2:
                ax.scatter(x, y, s=size, c='blue', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


G = 4  # Generation of Sierpinski triangle (keep small to start)
t = 1.0
U = -10.0  # Attractive interaction

H1, nodes, _ = tight_binding_sierpinski_2(G, t)
H2 = build_two_particle_H(H1, U)

# Solve for a few lowest eigenstates
num_states = len(nodes)**2-1
eigvals, eigvecs = eigsh(H2, k=num_states, which='SA')  # Smallest algebraic eigenvalues

# Plot the most bound state
plot_two_particle_state(eigvecs[:, 0], nodes, title=f"Bound State, E = {eigvals[0]:.3f}")

plt.figure(figsize=(8, 4))
plt.hist(eigvals, bins=50, density=True, edgecolor='k', alpha=0.7)
plt.xlabel('Energy')
plt.ylabel('Density of States')
plt.title(f'Density of States for U = {U}')
plt.grid(True)
plt.tight_layout()
plt.show()


def compute_continuum_gap(G, t, U, M=20):
    H1, nodes, _ = tight_binding_sierpinski_2(G, t)
    H2 = build_two_particle_H(H1, U)
    eigvals, _ = eigsh(H2, k=M, which='SA')
    eigvals = np.sort(eigvals)
    # continuum edge
    E_edge = -4*t
    # only bound‐state levels lie below E_edge
    bound_levels = eigvals[eigvals < E_edge]
    if len(bound_levels) == 0:
        return 0.0  # no bound state
    # highest bound level is the one closest to continuum
    highest_bound = bound_levels.max()
    return E_edge - highest_bound

gap_list = []
for Utest in np.linspace(-8,0,100):
    gap = compute_continuum_gap(G=3, t=1.0, U=Utest, M=20)
    gap_list.append(gap)
    print(f"U={Utest:>3}, continuum gap = {gap:.4f}")
        
plt.figure(figsize=(8, 4))
plt.plot(np.linspace(-8,0,100), gap_list)
plt.xlabel('U')
plt.ylabel('Gap width')
plt.title(f'Gap width against U')
plt.grid(True)
plt.tight_layout()
plt.show()


print("The eigenvalues are:", eigvals)
#eigvals, figs = plot_tb_sierpinski_2(G=1, k=1)