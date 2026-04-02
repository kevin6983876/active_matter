#!/usr/bin/env python3
"""
Twistor Fiber Bundle: PT = CP³ over spacetime, with CP¹ fibers.
Higgs field = choice of imaginary time direction on each fiber (arrow inside sphere).
Slider: 0 = symmetric (random arrows), 100 = broken symmetry (all arrows aligned).

Requires: pip install dash plotly
Run: python twistor_fiber_bundle.py  →  http://127.0.0.1:8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# -----------------------------------------------------------------------------
# Base manifold (spacetime): undulating 2D surface in 3D
# -----------------------------------------------------------------------------
N_BASE = 40
x_base = np.linspace(-2, 2, N_BASE)
y_base = np.linspace(-2, 2, N_BASE)
X_base, Y_base = np.meshgrid(x_base, y_base)
# Gentle undulation (curved spacetime)
Z_base = 0.3 * np.sin(0.8 * X_base) * np.cos(0.8 * Y_base)

# Sparse grid for fibers (5x5)
N_GRID = 5
x_fiber = np.linspace(-1.6, 1.6, N_GRID)
y_fiber = np.linspace(-1.6, 1.6, N_GRID)
# Fiber centers: project onto base surface (same undulation)
X_fiber, Y_fiber = np.meshgrid(x_fiber, y_fiber)
Z_fiber = 0.3 * np.sin(0.8 * X_fiber) * np.cos(0.8 * Y_fiber)

# Sphere (CP^1 fiber) parameters
SPHERE_RADIUS = 0.18
N_SPHERE_PHI = 20
N_SPHERE_THETA = 15
phi_s = np.linspace(0, 2 * np.pi, N_SPHERE_PHI)
theta_s = np.linspace(0, np.pi, N_SPHERE_THETA)
PHI, THETA = np.meshgrid(phi_s, theta_s)

def sphere_surface(cx, cy, cz, r):
    """Parametric sphere centered at (cx, cy, cz) with radius r."""
    x = cx + r * np.sin(THETA) * np.cos(PHI)
    y = cy + r * np.sin(THETA) * np.sin(PHI)
    z = cz + r * np.cos(THETA)
    return x, y, z

# Fixed random directions for slider=0 (reproducible)
np.random.seed(42)
n_centers = N_GRID * N_GRID
random_dirs = np.random.randn(n_centers, 3)
random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
# Aligned direction (e.g. "North Pole" = +Z in local fiber; we use global +Z for simplicity)
aligned_dir = np.array([0, 0, 1])

def slerp(d0, d1, t):
    """Spherical linear interpolation between unit vectors d0 and d1. t in [0,1]."""
    t = np.clip(t, 0, 1)
    dot = np.clip(np.dot(d0, d1), -1, 1)
    if dot > 0.9995:
        return (1 - t) * d0 + t * d1
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    d = s0 * d0 + s1 * d1
    return d / np.linalg.norm(d)

def make_arrows(break_param):
    """Arrow directions: break_param in [0,100] -> slerp(random, aligned, break_param/100)."""
    t = break_param / 100.0
    dirs = np.array([slerp(random_dirs[i], aligned_dir, t) for i in range(n_centers)])
    return dirs

def build_figure(break_param):
    arrow_dirs = make_arrows(break_param)
    centers = []
    for i in range(N_GRID):
        for j in range(N_GRID):
            cx = X_fiber[i, j]
            cy = Y_fiber[i, j]
            cz = Z_fiber[i, j]
            centers.append((cx, cy, cz))

    # Dark theme
    layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgb(18,18,24)",
        scene=dict(
            bgcolor="rgb(18,18,24)",
            xaxis=dict(backgroundcolor="rgb(18,18,24)", gridcolor="rgb(50,50,60)"),
            yaxis=dict(backgroundcolor="rgb(18,18,24)", gridcolor="rgb(50,50,60)"),
            zaxis=dict(backgroundcolor="rgb(18,18,24)", gridcolor="rgb(50,50,60)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig = go.Figure(layout=layout)

    # Base surface (spacetime)
    fig.add_trace(
        go.Surface(
            x=X_base, y=Y_base, z=Z_base,
            colorscale="Blues",
            opacity=0.5,
            showscale=False,
            name="Euclidean Spacetime (base)",
        )
    )

    # Fibers (CP^1 spheres) and arrows
    for k, (cx, cy, cz) in enumerate(centers):
        xs, ys, zs = sphere_surface(cx, cy, cz, SPHERE_RADIUS)
        fig.add_trace(
            go.Surface(
                x=xs, y=ys, z=zs,
                colorscale=[[0, "rgba(120,80,200,0.25)"], [1, "rgba(120,80,200,0.35)"]],
                opacity=0.4,
                showscale=False,
                name="Twistor fiber (CP¹)" if k == 0 else None,
            )
        )
        # Arrow: cone = arrowhead at tip, line = shaft from center to tip
        dx, dy, dz = arrow_dirs[k]
        arrow_len = SPHERE_RADIUS * 0.95
        tip_x, tip_y, tip_z = cx + arrow_len * dx, cy + arrow_len * dy, cz + arrow_len * dz
        # Cone: tip at (tip_x, tip_y, tip_z), base toward center (anchor="tip" => cone extends to tip - u,v,w)
        fig.add_trace(
            go.Cone(
                x=[tip_x], y=[tip_y], z=[tip_z],
                u=[-arrow_len * 0.5 * dx], v=[-arrow_len * 0.5 * dy], w=[-arrow_len * 0.5 * dz],
                anchor="tip",
                colorscale=[[0, "rgb(255,200,80)"], [1, "rgb(255,160,40)"]],
                showscale=False,
                name="Higgs (time dir.)" if k == 0 else None,
            )
        )
        # Shaft
        fig.add_trace(
            go.Scatter3d(
                x=[cx, tip_x], y=[cy, tip_y], z=[cz, tip_z],
                mode="lines",
                line=dict(color="rgb(255,180,60)", width=6),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text="Twistor Fiber Bundle: PT over Spacetime — Higgs as Imaginary Time<br>"
                 "<sub>Base: Euclidean Spacetime (HP¹ analog) · Spheres: Twistor Fibers (CP¹) · Arrows: Higgs (Imaginary Time Direction)</sub>",
            font=dict(size=14), x=0.5, xanchor="center",
        ),
    )

    return fig

# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Label("Symmetry Breaking / Time Evolution", style={"fontSize": "14px", "marginRight": "10px"}),
        dcc.Slider(0, 100, 5, value=0, id="symmetry-slider",
                   marks={0: "0 (Symmetric)", 50: "50", 100: "100 (Broken)"}),
    ], style={"padding": "20px", "maxWidth": "600px"}),
    dcc.Graph(id="twistor-bundle", figure=build_figure(0), style={"height": "75vh"}),
])

@app.callback(Output("twistor-bundle", "figure"), Input("symmetry-slider", "value"))
def update_figure(value):
    return build_figure(value)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
