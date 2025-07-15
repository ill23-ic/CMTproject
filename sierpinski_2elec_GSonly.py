import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sierpinski_1elec import build_sierpinski_2


class SierpinskiGSGetter:
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
        self.eigval, self.prob = self._gs_1body()

    def _gs_1body(self):
        """Literally just gives the 2 elec tight binding GS, as a one body wavefunction."""
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

        H = H_kin + H_int

        eigval, eigvec = eigsh(H, k=1, which='SA')
        eigvec = np.asarray(eigvec)  # Force dense conversion
        psi = eigvec[:, 0].reshape(self.N, self.N)
        prob_density = np.abs(psi)**2
        prob = np.array(np.sum(prob_density, axis=1)).flatten()

        return eigval, prob

    def plot_gs(self):
        """Plots the GS as above."""
        # Compute ONLY GS (k=1)
        eigval = self.eigval
        prob = self.prob

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


#sim = SierpinskiGSGetter(G=5, t=1.0, U=5.0)
#fig = sim.plot_gs()
#plt.show()
