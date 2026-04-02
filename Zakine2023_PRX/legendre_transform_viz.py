#!/usr/bin/env python3
"""
Legendre Transform: Geometric visualization

  L(v) = sup_θ [ vθ - H(θ) ]

- v: velocity (slider)
- θ: momentum (horizontal axis)
- H(θ): Hamiltonian (convex "bowl"); here H(θ) = 0.5 θ² + 1
- D(θ) = vθ - H(θ): vertical distance; maximum gives L(v)
- At optimum: θ* = v (from dD/dθ = 0), so L(v) = v·θ* - H(θ*) = 0.5v² - 1

Top: H(θ), line y = vθ, vertical gap at θ* (= L), tangent at θ* parallel to vθ.
Bottom: D(θ) with peak at (θ*, L) and horizontal reference at L.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------------------------------------------------------
# Hamiltonian and derived quantities
# -----------------------------------------------------------------------------
def H(theta):
    """Convex Hamiltonian (bowl): H(θ) = 0.5 θ² + 1."""
    return 0.5 * theta**2 + 1.0

def H_prime(theta):
    """dH/dθ = θ (for tangent line)."""
    return theta

def D(theta, v):
    """Vertical distance: D(θ) = vθ - H(θ). Maximized at θ* = v."""
    return v * theta - H(theta)

def L_from_v(v):
    """Legendre transform: L(v) = v·θ* - H(θ*) with θ* = v."""
    theta_star = v
    return v * theta_star - H(theta_star)  # = 0.5*v**2 - 1

def tangent_at(theta_star, theta_grid):
    """Tangent line to H at θ*: y = H(θ*) + H'(θ*)(θ - θ*)."""
    return H(theta_star) + H_prime(theta_star) * (theta_grid - theta_star)

# -----------------------------------------------------------------------------
# Build figure and axes
# -----------------------------------------------------------------------------
theta = np.linspace(-5, 5, 400)
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
plt.subplots_adjust(left=0.12, bottom=0.22, right=0.96, top=0.92, hspace=0.08)

# Fixed y-limits so the camera doesn't jump
ylim_top = (-10, 15)
ylim_bot = (-10, 10)
ax_top.set_ylim(ylim_top)
ax_bot.set_ylim(ylim_bot)
ax_top.set_xlim(-5, 5)
ax_bot.set_xlim(-5, 5)
ax_bot.set_xlabel(r"$\theta$ (momentum)")
ax_top.set_ylabel(r"$y$")
ax_bot.set_ylabel(r"$D(\theta) = v\theta - H(\theta)$")

# -----------------------------------------------------------------------------
# Static curves (updated in callback)
# -----------------------------------------------------------------------------
# Top: H(θ)
line_H, = ax_top.plot(theta, H(theta), color="C0", lw=2, label=r"$H(\theta)$")
# Line y = vθ (updated)
line_vtheta, = ax_top.plot(theta, 0 * theta, color="C1", lw=2, label=r"$v\theta$")
# Tangent at θ* (updated)
line_tangent, = ax_top.plot(theta, 0 * theta, "C2--", lw=1.5, label=r"Tangent at $\theta^*$")
# Vertical segment (gap L): from (θ*, vθ*) to (θ*, H(θ*))
vert_line = ax_top.plot([], [], "k-", lw=3, solid_capstyle="round")[0]
vert_gap_annotation = ax_top.annotate(
    r"$L = \sup_\theta D(\theta)$", xy=(0, 0), xytext=(0, 0),
    textcoords="data", fontsize=12, color="C3", fontweight="bold",
)
ax_top.axhline(0, color="gray", ls=":", alpha=0.6)
ax_top.axvline(0, color="gray", ls=":", alpha=0.6)
ax_top.legend(loc="upper center", framealpha=0.9)
ax_top.grid(True, alpha=0.3)

# Bottom: D(θ)
line_D, = ax_bot.plot(theta, D(theta, 2.0), color="C0", lw=2, label=r"$D(\theta)$")
peak_dot, = ax_bot.plot([], [], "ro", ms=12, zorder=5, label=r"$\sup_\theta D(\theta)$")
horiz_line, = ax_bot.plot([], [], "r--", lw=1.5, alpha=0.8)
ax_bot.axhline(0, color="gray", ls=":", alpha=0.6)
ax_bot.axvline(0, color="gray", ls=":", alpha=0.6)
ax_bot.legend(loc="upper right", framealpha=0.9)
ax_bot.grid(True, alpha=0.3)

title = fig.suptitle("", fontsize=12)

# -----------------------------------------------------------------------------
# Slider for v
# -----------------------------------------------------------------------------
ax_slider = plt.axes([0.2, 0.06, 0.6, 0.03])
slider_v = Slider(ax_slider, r"$v$ (velocity)", -4, 4, valinit=2.0, valstep=0.05)

def update(val):
    v = slider_v.val
    theta_star = v
    L = L_from_v(v)

    # Top: update vθ and tangent
    line_vtheta.set_ydata(v * theta)
    line_tangent.set_ydata(tangent_at(theta_star, theta))

    # Vertical gap at θ*
    y_line = v * theta_star
    y_H = H(theta_star)
    vert_line.set_data([theta_star, theta_star], [y_line, y_H])
    mid_y = (y_line + y_H) / 2
    vert_gap_annotation.set_position((theta_star, mid_y))
    vert_gap_annotation.set_text(r"$L = \sup_\theta D(\theta)$")

    # Bottom: D(θ) and peak
    line_D.set_ydata(D(theta, v))
    peak_dot.set_data([theta_star], [L])
    horiz_line.set_data([0, theta_star], [L, L])

    title.set_text(
        r"Legendre Transform: $v={:.2f}$, "
        r"Optimal $\theta^*={:.2f}$, $L={:.2f}$".format(v, theta_star, L)
    )
    # Add y-tick at L on bottom subplot
    old_ticks = list(ax_bot.get_yticks())
    new_ticks = sorted(set(old_ticks) | {L})
    ax_bot.set_yticks(new_ticks)
    labels = [
        r"$L = \sup_\theta D(\theta)$" if abs(t - L) < 1e-6 else f"{t:.1f}"
        for t in new_ticks
    ]
    ax_bot.set_yticklabels(labels)
    fig.canvas.draw_idle()

slider_v.on_changed(update)
update(slider_v.val)

plt.show()
