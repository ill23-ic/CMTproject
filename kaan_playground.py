#!/usr/bin/env python

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
#python /Users/kaanerman/Desktop/frank/CMTproject/kaan_playground.py

# Create a simple 4x4 sparse diagonal matrix as a placeholder for a Hubbard Hamiltonian
H = scipy.sparse.lil_matrix((4, 4))
H[0, 0] = 2
H[1, 1] = 3
H[2, 2] = 4
H[3, 3] = 5

# Convert to CSR format (required by eigsh)
H_csr = H.tocsr()

# Diagonalize: find the 2 smallest eigenvalues
eigvals, eigvecs = scipy.sparse.linalg.eigsh(H_csr, k=2, which='SA')

# Print results
print("Lowest 2 eigenvalues:", eigvals)
