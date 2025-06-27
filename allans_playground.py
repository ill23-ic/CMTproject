import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from functools import cached_property

class SimHubb2P:
    def __init__(self, L=10, t=1, U=0.01, periodic=True):
        self.L = L
        self.t = t
        self.U = U
        self.periodic = periodic

    @cached_property
    def hamiltonian(self):
        L, t, U = self.L, self.t, self.U
        
        # Single-particle hopping matrix
        # if change geometry of the lattice sites, change this
        i = np.arange(L)
        j_p = (i + 1) % L  # Periodic neighbors
        j_m = (i - 1) % L
        single_hop = sparse.lil_matrix((L, L))
        single_hop[i, j_p] = -t
        single_hop[i, j_m] = -t
        if not self.periodic:  # Open boundaries
            single_hop[0, -1] = 0
            single_hop[-1, 0] = 0
        
        # Two-particle kinetic term via tensor products, leave this unchanged
        I = sparse.eye(L)
        H_kin = sparse.kron(single_hop, I) + sparse.kron(I, single_hop)
        
        # we always keep interactions on-site
        H_int = sparse.diags([U if i == j else 0 
                         for i in range(L) 
                         for j in range(L)])
        
        return H_kin + H_int

    def eigen(self, k=6, return_vecs=True):
        """Returns k lowest eigenvalues/vectors"""
        H = self.hamiltonian
        eigvals, eigvecs = eigsh(H, k=k, which='SA')
        eigvecs = np.asarray(eigvecs)  # Force dense conversion
        return (eigvals, eigvecs) if return_vecs else eigvals

    def plot_eigenstate(self, state_idx=0):
        """Plot probability density of specified eigenstate"""
        # Get enough states (minimum 6 or state_idx+1)
        k = max(6, state_idx + 1)
        eigvals, eigvecs = self.eigen(k=k)
        
        if state_idx >= eigvecs.shape[1]:
            raise ValueError(
                f"state_idx {state_idx} >= available states ({eigvecs.shape[1]})"
            )

        psi = eigvecs[:, state_idx].reshape(self.L, self.L)
        prob_density = np.abs(psi)**2

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Marginal probabilities
        sites = np.arange(self.L)
        ax1.bar(sites - 0.2, np.sum(prob_density, axis=1), width=0.4, label='Up')
        ax1.bar(sites + 0.2, np.sum(prob_density, axis=0), width=0.4, label='Down')
        ax1.set(xlabel='Site', ylabel='Probability', 
                title=f'State {state_idx} (E={eigvals[state_idx]:.3f})')
        ax1.legend()

        # 2. Two-particle density
        im = ax2.imshow(prob_density, cmap='viridis', origin='lower',
                        extent=[0, self.L, 0, self.L])
        ax2.set(xlabel='Down pos', ylabel='Up pos', 
                title='Joint Probability')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        return fig
