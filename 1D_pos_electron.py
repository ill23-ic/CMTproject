import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import eigsh

def hubbard_1d_two_particles(L, t, U):
    """
    Construct and diagonalize 1D Hubbard Hamiltonian for two particles with opposite spins
    
    Parameters:
    L : int - number of lattice sites
    t : float - hopping amplitude
    U : float - on-site interaction strength
    
    Returns:
    eigenvalues : array - energy eigenvalues
    eigenvectors : array - corresponding eigenvectors
    """
    
    # Single-particle Hamiltonian (same for both spins)
    # Hopping term (-t sum_{<i,j>} c_i^† c_j)
    single_hop = diags([-t]*L, [-1]) + diags([-t]*L, [1])
    single_hop = single_hop.tocsc()  # convert to compressed sparse column format
    
    # Identity matrix for single-particle space
    I = identity(L, format='csc')
    
    # Two-particle Hamiltonian:
    # H = H_hop ⊗ I + I ⊗ H_hop + U sum_i n_{i,up} n_{i,down}
    
    # Hopping terms
    H_hop = kron(single_hop, I) + kron(I, single_hop)
    
    # On-site interaction term
    # This is U on states where both particles are on the same site
    interaction = np.zeros((L, L))
    for i in range(L):
        interaction[i, i] = U
    
    # Convert to sparse and diagonalize
    H_total = H_hop + interaction
    
    # Diagonalize (using sparse diagonalization for large L)
    eigenvalues, eigenvectors = eigsh(H_total, k=min(10, L**2-2), which='SA')
    
    return eigenvalues, eigenvectors

# Example usage
L = 4  # number of sites
t = 1.0  # hopping parameter
U = 2.0  # on-site interaction

eigenvalues, eigenvectors = hubbard_1d_two_particles(L, t, U)

print("Energy eigenvalues:")
print(eigenvalues)
