#!/usr/bin/env python3
"""
4D Double Rotation → Stereographic Projection to 3D
Multiple fixed cases: no slider; plot several ω₂/ω₁ ratios side by side.

Physics:
  Point on S³ (R₁=R₂=1/√2) rotating in (x,y) at ω₁ and (z,w) at ω₂.
  Stereographic projection (x,y,z,w) → (X,Y,Z) = (x,y,z)/(1-w).
  At ω₂ = ω₁ (isoclinic) the path collapses to a Villarceau circle in 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
R1 = R2 = 1.0 / np.sqrt(2)
OMEGA_1 = 1.0
T_MAX = 40 * np.pi
N_POINTS = 5000
t = np.linspace(0, T_MAX, N_POINTS)

def stereographic(x, y, z, w):
    denom = 1.0 - w
    denom = np.where(np.abs(denom) < 0.08, np.sign(denom) * 0.08, denom)
    return x / denom, y / denom, z / denom

def compute_trajectory(omega2):
    x = R1 * np.cos(OMEGA_1 * t)
    y = R1 * np.sin(OMEGA_1 * t)
    z = R2 * np.cos(omega2 * t)
    w = R2 * np.sin(omega2 * t)
    return stereographic(x, y, z, w)

def get_title(omega2):
    if np.isclose(omega2, 1.0, atol=0.02):
        return r"Isoclinic ($\omega_2=1$): Perfect Circle"
    if np.isclose(omega2, 1.5, atol=0.02) or np.isclose(omega2, 2.0, atol=0.02) or \
       np.isclose(omega2, 0.5, atol=0.02):
        return r"Torus Knot ($\omega_2={:.1f}$, rational)".format(omega2)
    return r"Dense Coil ($\omega_2={:.3g}$, irrational)".format(omega2)

# -----------------------------------------------------------------------------
# Cases to plot: (ω₂, label for subplot title)
# -----------------------------------------------------------------------------
cases = [
    (1.0,   "Isoclinic: Perfect Circle"),
    (1.5,   "Torus Knot (ω₂=1.5)"),
    (2.0,   "Torus Knot (ω₂=2)"),
    (0.5,   "Torus Knot (ω₂=0.5)"),
    (1.414, "Dense Coil (ω₂≈√2)"),
    (np.e / 2, "Dense Coil (ω₂≈e/2)"),
]
n_cases = len(cases)
n_cols = 3
n_rows = (n_cases + n_cols - 1) // n_cols

fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
for i, (omega2, label) in enumerate(cases):
    ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
    X, Y, Z = compute_trajectory(omega2)
    ax.plot(X, Y, Z, color="steelblue", lw=0.7, alpha=0.9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("$X$", fontsize=9)
    ax.set_ylabel("$Y$", fontsize=9)
    ax.set_zlabel("$Z$", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(get_title(omega2), fontsize=10)

fig.suptitle(r"4D double rotation → stereographic projection to 3D ($\omega_1=1$, various $\omega_2$)", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("isoclinic_stereographic_cases.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: isoclinic_stereographic_cases.png")
