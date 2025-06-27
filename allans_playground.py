import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


class SimHubb2P:
    def __init__(self, L=10, t=1, U=0.01, periodic=True):
        self.L = L
        self.t = t
        self.U = U
        self.periodic = periodic

    def hamiltonian(self):
        # Single-particle hopping matrix
        # if change geometry of the lattice sites, change this
        L = self.L
        U = self.U
        t = self.t
        single_hop = np.zeros((L, L))
        for i in range(L):
            for delta in [-1, 1]:  # nearest neighbors
                j = (i + delta) % L if self.periodic else i + delta
                if 0 <= j < L:  # Valid index
                    single_hop[i, j] = -t

        # Two-particle kinetic term via tensor products, leave this unchanged
        H_kin = np.kron(single_hop, np.eye(L)) + np.kron(np.eye(L), single_hop)

        # we always keep interactions on-site
        H_int = np.zeros((L*L, L*L))
        np.fill_diagonal(H_int, [U if i//L == i %
                         L else 0 for i in range(L*L)])

        return H_int+H_kin

    def eigen(self, return_vecs=True):
        H = self.hamiltonian()
        eigvals, eigvecs = eigh(H)
        if return_vecs is True:
            return eigvals, eigvecs
        else:
            return eigvals

    def plot_eigenstate(self, state_idx=0):
        """
        Plot the probability density of an eigenstate on the lattice

        Args:
            state_idx (int): Index of eigenstate to plot (0 = ground state)

        Raises:
            ValueError: If state_idx exceeds Hilbert space dimension
        """
        # Check if state_idx is valid
        max_states = self.L**2
        if state_idx >= max_states:
            raise ValueError(
                f"state_idx must be < {max_states} (L²) for L={self.L}")

        # Get the eigenstate
        _, eigvecs = self.eigen()
        psi = eigvecs[:, state_idx].reshape(self.L, self.L)

        # Calculate probability densities
        prob_density = np.abs(psi)**2
        prob_up = np.sum(prob_density, axis=1)  # Marginal for up electron
        prob_down = np.sum(prob_density, axis=0)  # Marginal for down electron

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot individual particle probabilities
        sites = np.arange(self.L)
        ax1.bar(sites - 0.2, prob_up, width=0.4,
                label='Up electron', alpha=0.7)
        ax1.bar(sites + 0.2, prob_down, width=0.4,
                label='Down electron', alpha=0.7)
        ax1.set_xlabel('Lattice site')
        ax1.set_ylabel('Probability density')
        ax1.set_title(
            f'Individual particle probabilities\n(Eigenstate {state_idx})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot two-particle density
        im = ax2.imshow(prob_density, origin='lower', cmap='viridis',
                        extent=[0, self.L, 0, self.L])
        ax2.set_xlabel('Down electron position')
        ax2.set_ylabel('Up electron position')
        ax2.set_title('Two-particle probability density')
        plt.colorbar(im, ax=ax2, label='Probability')

        plt.tight_layout()
        return fig
