#!/usr/bin/env python3
"""
非平衡相圖：打破細緻平衡 (Broken Detailed Balance)

Maier-Stein 型 2D 動力學：
  dx/dt = x - x³ - β x y²
  dy/dt = -(1 + x²) y

對比平衡態的異宿軌道（沿 x 軸）與非平衡的 Forward/Backward MAP 路徑。
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 參數與網格
# =============================================================================
beta = 10.0
x = np.linspace(-1.5, 1.5, 80)
y = np.linspace(-1.0, 1.0, 60)
X, Y = np.meshgrid(x, y)

# 向量場
u = X - X**3 - beta * X * Y**2
v = -(1 + X**2) * Y
magnitude = np.sqrt(u**2 + v**2)

# =============================================================================
# 畫布與背景（向量場強度）
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
# 背景：越深色代表力場越強（歸一化後用 Gray_r：大→黑）
mag_norm = magnitude / (magnitude.max() + 1e-8)
ax.pcolormesh(X, Y, mag_norm, shading="auto", cmap="Greys_r", alpha=0.6)
# 流線
strm = ax.streamplot(X[0], Y[:, 0], u, v, color=magnitude, cmap="viridis", linewidth=1.2, density=1.5)
fig.colorbar(strm.lines, ax=ax, label=r"$|\mathbf{v}|$")

# =============================================================================
# 固定點
# =============================================================================
ax.plot(-1, 0, "ko", markersize=14, label="Stable node", zorder=5)
ax.plot(1, 0, "ko", markersize=14, zorder=5)
ax.plot(0, 0, "kx", markersize=14, markeredgewidth=3, label="Saddle", zorder=5)

# =============================================================================
# 概念性路徑
# =============================================================================
# 平衡態異宿軌道：沿 x 軸
x_het = np.linspace(-1, 1, 100)
y_het = np.zeros_like(x_het)
ax.plot(x_het, y_het, "r--", linewidth=2, label="Heteroclinic (detailed balance)")

# 非平衡 Forward MAP：(-1,0) → (1,0)，上半部凸起 y = 0.5*(1 - x²)
x_fwd = np.linspace(-1, 1, 80)
y_fwd = 0.5 * (1 - x_fwd**2)
ax.plot(x_fwd, y_fwd, color="darkblue", linewidth=2.5, label="Forward MAP")
# 方向箭頭（沿曲線取幾點）
for i in [20, 40, 60]:
    dx = x_fwd[i+1] - x_fwd[i]
    dy = y_fwd[i+1] - y_fwd[i]
    ax.annotate("", xy=(x_fwd[i]+dx*0.5, y_fwd[i]+dy*0.5),
                xytext=(x_fwd[i], y_fwd[i]),
                arrowprops=dict(arrowstyle="->", color="darkblue", lw=2))

# 非平衡 Backward MAP：(1,0) → (-1,0)，下半部凸起 y = -0.5*(1 - x²)
x_bwd = np.linspace(1, -1, 80)  # 從 1 到 -1
y_bwd = -0.5 * (1 - x_bwd**2)
ax.plot(x_bwd, y_bwd, color="orangered", linewidth=2.5, label="Backward MAP")
for i in [20, 40, 60]:
    dx = x_bwd[i+1] - x_bwd[i]
    dy = y_bwd[i+1] - y_bwd[i]
    ax.annotate("", xy=(x_bwd[i]+dx*0.5, y_bwd[i]+dy*0.5),
                xytext=(x_bwd[i], y_bwd[i]),
                arrowprops=dict(arrowstyle="->", color="orangered", lw=2))

# =============================================================================
# 美化
# =============================================================================
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Nonequilibrium Phase Transition: Broken Detailed Balance")
plt.tight_layout()
plt.savefig("maier_stein_phase_portrait.png", dpi=150, bbox_inches="tight")
plt.show()
