import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#yo_2 #hello first commit vhs
def compute_K0_block(L, t, U):
    K_index = 0
    momenta = 2 * np.pi * np.arange(L) / L  # Allowed momenta
    basis = []
    for n1 in range(L):
        n2 = (K_index - n1) % L
        basis.append(('↑↓', n1, n2))
    basis_size = len(basis)
    H_block = np.zeros((basis_size, basis_size))
    
    # Kinetic energy contribution
    for i in range(basis_size):
        sector, n1, n2 = basis[i]
        p1 = momenta[n1]
        p2 = momenta[n2]
        kinetic = -2 * t * (np.cos(p1) + np.cos(p2))
        H_block[i, i] = kinetic
    
    # Interaction contribution
    interaction = -U / L * np.ones((basis_size, basis_size))
    H_block += interaction
    
    eigenvalues = np.linalg.eigvalsh(H_block)
    return eigenvalues

# Parameters
L = 200
t = 1
U_values = np.linspace(0.001, 1, 900)  # Adjust U range as needed
energy_differences = []

# Compute energy gaps for each U
for U in U_values:
    evals = compute_K0_block(L, t, U)
    if len(evals) >= 2:
        gap = evals[1] - evals[0]
        energy_differences.append(gap)
    else:
        energy_differences.append(0)

# Define an empirical quadratic fit function (no theory assumed)
def empirical_fit(U, a, b, c):
    return a * U**2 + b * U + c  # Quadratic: aU² + bU + c

# Fit the data to the empirical function
params, covariance = curve_fit(empirical_fit, U_values, energy_differences)
a, b, c = params
print(f"Empirical fit coefficients: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

# Generate the fitted curve
fitted_gap = empirical_fit(U_values, a, b, c)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(U_values, energy_differences, s=5, alpha=0.6, label='Numerical Data')
plt.plot(U_values, fitted_gap, 'r-', linewidth=2, label=f'Empirical Fit: ${a:.4f}U^2 + {b:.4f}U + {c:.4f}$')
plt.xlabel('Interaction Strength $U$', fontsize=12)
plt.ylabel('Energy Gap', fontsize=12)
plt.title('Empirical Fit of Energy Gap at $K=0$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
