import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def construct_band_gaps_vs_U(L, t, U_values):
    gap_data = []
    momenta_x = 2 * np.pi * np.arange(L) / L
    momenta_y = 2 * np.pi * np.arange(L) / L

    for U in U_values:
        min_gap = np.inf
        min_gap_KxKy = (None, None)

        for Kx_index in range(L):
            for Ky_index in range(L):
                Kx = 2 * np.pi * Kx_index / L
                Ky = 2 * np.pi * Ky_index / L
                basis = []

                for n1x in range(L):
                    for n1y in range(L):
                        n2x = (Kx_index - n1x) % L
                        n2y = (Ky_index - n1y) % L
                        basis.append((n1x, n1y, n2x, n2y))

                basis_size = len(basis)
                if basis_size < 2:
                    continue

                H_block = np.zeros((basis_size, basis_size))

                for i, (n1x, n1y, n2x, n2y) in enumerate(basis):
                    px1 = momenta_x[n1x]
                    py1 = momenta_y[n1y]
                    px2 = momenta_x[n2x]
                    py2 = momenta_y[n2y]
                    kinetic = -2 * t * (np.cos(px1) + np.cos(py1) + np.cos(px2) + np.cos(py2))
                    H_block[i, i] = kinetic

                H_block += -U / (L**2) * np.ones((basis_size, basis_size))
                eigenvalues = np.linalg.eigvalsh(H_block)
                if len(eigenvalues) >= 2:
                    gap = eigenvalues[1] - eigenvalues[0]
                    if gap < min_gap:
                        min_gap = gap
                        min_gap_KxKy = (Kx, Ky)

        gap_data.append((U, min_gap, *min_gap_KxKy))

    return np.array(gap_data)

# Parameters
L = 5
t = 1
U_values = np.linspace(0, 1, 15)

# Get gap data
gap_data = construct_band_gaps_vs_U(L, t, U_values)
U_vals = gap_data[:, 0]
gaps = gap_data[:, 1]

# Quadratic fit
def poly2(U, a, b, c):
    return a * U**2 + b * U + c

params_poly2, _ = curve_fit(poly2, U_vals, gaps)
fit_poly2 = poly2(U_vals, *params_poly2)
eqn_poly2 = f"{params_poly2[0]:.2e}·U² + {params_poly2[1]:.2e}·U + {params_poly2[2]:.2e}"

# 4th-degree polynomial fit
def poly4(U, a, b, c, d, e):
    return a * U**4 + b * U**3 + c * U**2 + d * U + e

params_poly4, _ = curve_fit(poly4, U_vals, gaps)
fit_poly4 = poly4(U_vals, *params_poly4)
eqn_poly4 = (
    f"{params_poly4[0]:.2e}·U⁴ + {params_poly4[1]:.2e}·U³ + "
    f"{params_poly4[2]:.2e}·U² + {params_poly4[3]:.2e}·U + {params_poly4[4]:.2e}"
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(U_vals, gaps, 'o', label="Computed Gaps", alpha=0.5)
plt.plot(U_vals, fit_poly2, label=f"Quadratic Fit:\n{eqn_poly2}", linewidth=2)
plt.plot(U_vals, fit_poly4, label=f"4th Degree Fit:\n{eqn_poly4}", linewidth=2)

plt.xlabel("Interaction Strength $U$")
plt.ylabel("Minimum Energy Gap $\Delta E$")
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
