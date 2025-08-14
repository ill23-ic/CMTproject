import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from functools import cached_property
from sierpinski_1elec import build_sierpinski_2


class FractalHubbard2P:
    def __init__(self, G=3, t=1.0, U=0.01):
        """
        Two-electron Hubbard model on Sierpiński fractal
        Args:
            G: Generation of Sierpiński triangle
            t: Hopping amplitude
            U: On-site interaction strength
        """
        self.G = G
        self.t = t
        self.U = U
        self.nodes, self.edges = build_sierpinski_2(G)
        self.N = len(self.nodes)  # Number of sites

    @cached_property
    def hamiltonian(self):
        """Constructs two-electron Hamiltonian with on-site interaction"""
        # Single-particle hopping matrix
        single_hop = sparse.lil_matrix((self.N, self.N))
        for u, v in self.edges:
            single_hop[u, v] = -self.t
            single_hop[v, u] = -self.t

        # Two-particle kinetic term
        Id = sparse.eye(self.N)
        H_kin = sparse.kron(single_hop, Id) + sparse.kron(Id, single_hop)

        # On-site interaction (U only when both electrons are on same site)
        H_int = sparse.diags([self.U if i == j else 0
                              for i in range(self.N)
                              for j in range(self.N)])

        return H_kin + H_int

    def eigen(self, k=6, return_vecs=True):
        """Returns k lowest eigenvalues/vectors"""
        H = self.hamiltonian
        eigvals, eigvecs = eigsh(H, k=k, which='SA')
        eigvecs = np.asarray(eigvecs)  # Force dense conversion
        return (eigvals, eigvecs) if return_vecs else eigvals

    def plot_eigenstate(self, state_idx=0):
        """Visualizes two-electron eigenstate with full correlation information"""
        eigvals, eigvecs = self.eigen(k=state_idx+1)
        psi = eigvecs[:, state_idx].reshape(self.N, self.N)
        prob_density = np.abs(psi)**2

        fig = plt.figure(figsize=(16, 6))

        # 1. Fractal lattice with single-electron density
        ax1 = fig.add_subplot(131)
        # Probability for electron 1
        marginal_prob = np.sum(prob_density, axis=1)
        nodes = np.array(self.nodes)

        # Size/color shows probability density
        sc = ax1.scatter(nodes[:, 0], nodes[:, 1], c=marginal_prob,
                         s=200*marginal_prob/np.max(marginal_prob),
                         cmap='viridis')

        # Draw edges
        for u, v in self.edges:
            ax1.plot([nodes[u, 0], nodes[v, 0]], [nodes[u, 1], nodes[v, 1]],
                     'k-', alpha=0.2)

        ax1.set_title(f"Electron 1 Density\n(E={eigvals[state_idx]:.3f})")
        plt.colorbar(sc, ax=ax1, label='Probability')
        ax1.set_aspect('equal')

        # 2. Fractal lattice with double occupancy
        ax2 = fig.add_subplot(132)
        double_occ = np.diag(prob_density)
        sc2 = ax2.scatter(nodes[:, 0], nodes[:, 1], c=double_occ,
                          s=200*double_occ/np.max(double_occ+1e-10),
                          cmap='plasma')

        for u, v in self.edges:
            ax2.plot([nodes[u, 0], nodes[v, 0]], [nodes[u, 1], nodes[v, 1]],
                     'k-', alpha=0.2)

        ax2.set_title("Double Occupancy Probability")
        plt.colorbar(sc2, ax=ax2, label='P(x₁=x₂)')
        ax2.set_aspect('equal')

        # 3. Full correlation matrix
        ax3 = fig.add_subplot(133)
        im = ax3.imshow(prob_density, cmap='viridis', origin='lower',
                        extent=[0, self.N, 0, self.N])
        ax3.set(xlabel='Electron 2 Position', ylabel='Electron 1 Position',
                title='Joint Probability Density')
        plt.colorbar(im, ax=ax3, label='P(x₁,x₂)')

        plt.tight_layout()
        return fig

    def plot_ground_state_fast(self):
        """Optimized for ground-state only, for visuals. No edge plots."""
        # Compute ONLY GS (k=1)
        eigval, eigvec = eigsh(self.hamiltonian, k=1, which='SA')
        psi = eigvec[:, 0].reshape(self.N, self.N)
        prob_density = np.abs(psi)**2
        prob = np.array(np.sum(prob_density, axis=1)).flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        nodes = np.array(self.nodes)

        norm = plt.Normalize(0, prob.max())
        colors = plt.cm.viridis(norm(prob))

        ax.bar3d(nodes[:, 0], nodes[:, 1], np.zeros_like(prob),
                 0.02, 0.02, prob, color=colors, edgecolor='k')

        ax.set_title(f"GS Probability Density (E={eigval[0]:.3f})")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                     ax=ax, label='Probability')
        return fig

    def bound_state_gap(self):
        """
        Takes 2*(no of sites) lowest-energy states and computes bound state gap (if any) wrt given information.
        Bound state gap= E_continuum_edge - E_highest_bound
        """

        N = 2*len(self.nodes)
        M = max(20, N)
        eigvals = self.eigen(M, return_vecs=False)
        eigvals = np.sort(eigvals)

        E_edge = -4*self.t
        bound_levels = eigvals[eigvals < E_edge]
        if len(bound_levels) == 0:
            return 0.0  # no bound state
        # highest bound level is the one closest to continuum
        highest_bound = bound_levels.max()
        return E_edge - highest_bound
    
    def count_bound_states(self, M_scale=2.0):
        N = len(self.nodes)
        M = max(20, int(M_scale * N))  # Ensure enough states
        eigvals = np.sort(self.eigen(M, return_vecs=False))
        E_edge = -4 * self.t
        bound_levels = eigvals[eigvals < E_edge]
        return len(bound_levels)  # Number of bound states

#Example implementations

# Strong interaction case
# sim = FractalHubbard2P(G=5, t=1.0, U=5.0)
# fig = sim.plot_ground_state_fast()  # Ground state
# plt.show()

# Compare to weak interaction
# sim_weak = FractalHubbard2P(G=5, t=1.0, U=0.1)
# fig = sim_weak.plot_eigenstate_3d(0)
# plt.show()

gap_list = []
for Utest in np.linspace(-8,0,100):
    system = FractalHubbard2P(G=3, t=1.0, U=Utest)
    gap = system.bound_state_gap()
    gap_list.append(gap)
        
plt.figure(figsize=(8, 4))
plt.plot(np.linspace(-8,0,100), gap_list)
plt.xlabel('U')
plt.ylabel('Gap width')
plt.title('Gap width against U')
plt.grid(True)
plt.tight_layout()
plt.show()
