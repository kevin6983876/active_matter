#!/usr/bin/env python3
"""
模組一：時空的盡頭是起點 —— 共形緊緻化 (R⁴ → S⁴)
The Compactification Sandbox: 2D 平面捲成 3D 黎曼球 (Riemann Sphere)

對應論文：Conformally compactifying R⁴ to S⁴, HP¹...
直覺：用 2D 棋盤格平面 + 粒子，經滑桿「拉拔、捲曲」成封閉 3D 球體，
      觀察原本無限遠的粒子在北極交匯（球極平面投影 Stereographic Projection）。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import font_manager

# 偵測是否有可顯示中文的字型；若沒有則用英文標籤避免亂碼
def _setup_cjk_or_english():
    """若系統有 CJK 字型則設定，否則回傳 False 讓程式改用英文。"""
    cjk_candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft JhengHei",
        "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in cjk_candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + [
                x for x in plt.rcParams["font.sans-serif"] if x != name
            ]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

_use_chinese = _setup_cjk_or_english()

# 圖上顯示的文字（有 CJK 用中文，否則英文）
if _use_chinese:
    SLIDER_LABEL = r"緊緻化 $t$ (平面 → 球)"
    TITLE = r"模組一：共形緊緻化 — 平面 $\mathbb{R}^2$ 捲成黎曼球 $S^2$"
    NORTH_LABEL = r"北極 (無窮遠 $\leftrightarrow$ 此點)"
else:
    SLIDER_LABEL = r"Compactification $t$ (plane → sphere)"
    TITLE = r"Module 1: Conformal compactification — $\mathbb{R}^2$ → $S^2$ (Riemann sphere)"
    NORTH_LABEL = r"North pole (infinity $\leftrightarrow$ here)"

# -----------------------------------------------------------------------------
# 球極平面投影：平面 (u,v) ↔ 單位球面 S²（無窮遠 ↔ 北極）
# Inverse: (u,v) -> (x,y,z) on unit sphere,  (u,v)->∞ -> North pole (0,0,1)
# -----------------------------------------------------------------------------

def plane_to_sphere(u, v):
    """Inverse stereographic: (u,v) in R² -> (x,y,z) on unit S²."""
    r2 = u**2 + v**2
    denom = 1.0 + r2
    x = 2*u / denom
    y = 2*v / denom
    z = (r2 - 1) / denom
    return x, y, z


def morph_point(u, v, t, plane_scale=1.0):
    """
    t=0: 平面 (u/scale, v/scale, 0)
    t=1: 球面 inverse_stereographic(u,v)
    線性插值得到「拉拔、捲曲」過程。
    """
    x_plane = u / plane_scale
    y_plane = v / plane_scale
    z_plane = 0.0
    x_s, y_s, z_s = plane_to_sphere(u, v)
    x = (1 - t) * x_plane + t * x_s
    y = (1 - t) * y_plane + t * y_s
    z = (1 - t) * z_plane + t * z_s
    return x, y, z


# -----------------------------------------------------------------------------
# 參數：平面顯示範圍、棋盤格、粒子位置
# -----------------------------------------------------------------------------
L = 4.0              # 平面 (u,v) 範圍 [-L, L]
plane_scale = 2.0    # 平面縮放，使與單位球視覺相當
grid_step = 1.0      # 棋盤格間距
particles_uv = [     # 粒子在平面上的 (u,v)，最後一個代表「無窮遠」
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, -2),
    (2, 2),
    (5, 5),   # 遠處 → 北極
]
# 最後一顆粒子用綠色標示「無窮遠 → 北極」
particle_colors = ["coral"] * (len(particles_uv) - 1) + ["green"]

# 棋盤格線：固定 u 的線 + 固定 v 的線
u_vals = np.arange(-L, L + grid_step/2, grid_step)
v_vals = np.arange(-L, L + grid_step/2, grid_step)
n_pts = 80  # 每條線的取樣點數

def make_grid_lines():
    """生成格線的 (u,v) 線段，用於後續 morph 繪製。"""
    lines = []
    for u in u_vals:
        v_line = np.linspace(-L, L, n_pts)
        lines.append(np.column_stack([np.full_like(v_line, u), v_line]))
    for v in v_vals:
        u_line = np.linspace(-L, L, n_pts)
        lines.append(np.column_stack([u_line, np.full_like(u_line, v)]))
    return lines


# -----------------------------------------------------------------------------
# 繪圖：3D 軸 + 滑桿
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 滑桿：緊緻化參數 t ∈ [0, 1]
ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
slider = Slider(ax_slider, SLIDER_LABEL, 0, 1, valinit=0, valstep=0.02)

# 儲存繪圖物件以便更新
grid_line_artists = []
particle_artist = None
sphere_wireframe = None


def init_plot():
    """建立格線與粒子的 Line3D / scatter 物件。"""
    global grid_line_artists, particle_artist, sphere_wireframe
    grid_lines = make_grid_lines()
    t0 = slider.val

    for pts in grid_lines:
        uu, vv = pts[:, 0], pts[:, 1]
        xx, yy, zz = morph_point(uu, vv, t0, plane_scale)
        line, = ax.plot(xx, yy, zz, color="steelblue", alpha=0.6, lw=0.8)
        grid_line_artists.append(line)

    # 粒子（橘色；最後一顆綠色＝無窮遠→北極）
    pu = [p[0] for p in particles_uv]
    pv = [p[1] for p in particles_uv]
    px, py, pz = morph_point(np.array(pu), np.array(pv), t0, plane_scale)
    particle_artist = ax.scatter(
        px, py, pz, c=particle_colors, s=80, depthshade=True, zorder=10
    )
    if not hasattr(init_plot, "_particle_artist_ref"):
        init_plot._particle_artist_ref = [particle_artist]
    else:
        init_plot._particle_artist_ref[0] = particle_artist

    # 可選：單位球線框當參考（半透明）
    phi = np.linspace(0, np.pi, 20)
    th = np.linspace(0, 2 * np.pi, 30)
    PHI, TH = np.meshgrid(phi, th)
    sx = np.sin(PHI) * np.cos(TH)
    sy = np.sin(PHI) * np.sin(TH)
    sz = np.cos(PHI)
    sphere_wireframe = ax.plot_wireframe(sx, sy, sz, color="gray", alpha=0.15, linewidth=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(TITLE)
    set_axis_limits(ax, t0)


def set_axis_limits(ax, t):
    r = 1.5
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect([1, 1, 1])


def update(t):
    t = float(t)
    grid_lines = make_grid_lines()
    for i, pts in enumerate(grid_lines):
        uu, vv = pts[:, 0], pts[:, 1]
        xx, yy, zz = morph_point(uu, vv, t, plane_scale)
        grid_line_artists[i].set_data_3d(xx, yy, zz)

    pu = np.array([p[0] for p in particles_uv])
    pv = np.array([p[1] for p in particles_uv])
    px, py, pz = morph_point(pu, pv, t, plane_scale)
    if hasattr(init_plot, "_particle_artist_ref") and init_plot._particle_artist_ref:
        sc = init_plot._particle_artist_ref[0]
        sc._offsets3d = (px, py, pz)

    if hasattr(update, "_north_label") and update._north_label is not None:
        update._north_label.set_visible(t > 0.7)
    fig.canvas.draw_idle()


def on_slider_change(val):
    update(val)


slider.on_changed(on_slider_change)
init_plot()
update._north_label = ax.text(0, 0, 1.15, NORTH_LABEL, fontsize=9, color="green")
update._north_label.set_visible(False)

plt.subplots_adjust(bottom=0.08)
plt.show()
