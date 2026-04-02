#!/usr/bin/env python3
"""
Active Phase Separation：三種漲落視覺化

1x3 子圖說明為何在 Classical Nucleation Theory (CNT) 中，
只有「平均半徑漲落」是重要的反應座標，
而「形狀漲落」與「介面剖面漲落」可被忽略。
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 畫布：前兩格極座標，第三格直角座標
# =============================================================================
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1, projection="polar")
ax2 = fig.add_subplot(1, 3, 2, projection="polar")
ax3 = fig.add_subplot(1, 3, 3)
axes = [ax1, ax2, ax3]

# 共用極座標角度
theta = np.linspace(0, 2 * np.pi, 200)
R = 5.0

# =============================================================================
# 圖一：平均半徑漲落 R_t（CNT 中唯一重要的）
# =============================================================================
ax1.plot(theta, np.ones_like(theta) * R, "k-", linewidth=2, label=r"$R=5$")
ax1.plot(theta, np.ones_like(theta) * 4, "b--", linewidth=1.5, label=r"$R=4$")
ax1.plot(theta, np.ones_like(theta) * 6, "r--", linewidth=1.5, label=r"$R=6$")
# 雙向箭頭標示 ΔR_t（沿 θ=0 方向）
ax1.annotate(
    "",
    xy=(0, 6),
    xytext=(0, 4),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2),
)
ax1.text(0.15, 5.0, r"$\Delta R_t$", fontsize=14, ha="left")
ax1.set_title(
    "(i) Mean Radius $R_t$\n(Relevant: Slow Reaction Coord)",
    pad=20,
)
ax1.set_ylim(0, 8)
# 隱藏極座標刻度，保持乾淨
ax1.set_xticklabels([])
ax1.set_yticklabels([])

# =============================================================================
# 圖二：形狀漲落 δR（毛細波，被 σ_cw 壓制）
# =============================================================================
ax2.plot(theta, np.ones_like(theta) * R, "gray", linestyle="--", linewidth=1.5, label=r"$R=5$")
r_wavy = 5 + 0.6 * np.sin(8 * theta)
ax2.plot(theta, r_wavy, color="orangered", linewidth=2, label=r"$r(\theta)=R+\delta R$")
# 紅色小箭頭指向圓心，表示表面張力將變形拉回
n_arrows = 5
theta_arrows = np.linspace(0, 2 * np.pi, n_arrows, endpoint=False)
for th in theta_arrows:
    r_at = 5 + 0.6 * np.sin(8 * th)
    ax2.annotate(
        "",
        xy=(th, r_at - 0.6),
        xytext=(th, r_at),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )
ax2.set_title(
    r"(ii) Shape Fluctuation $\delta R$" + "\n(Irrelevant: Suppressed by $\\sigma_{cw}$)",
    pad=20,
)
ax2.set_ylim(0, 8)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

# =============================================================================
# 圖三：介面剖面漲落（快速擴散鬆弛，可忽略）
# =============================================================================
r_1d = np.linspace(2, 8, 500)
R_1d = 5.0
w0 = 0.5
# 基準介面剖面
phi0 = 0.5 * (1 - np.tanh((r_1d - R_1d) / w0))
# 剖面漲落：介面變窄 / 變寬
phi_narrow = 0.5 * (1 - np.tanh((r_1d - R_1d) / 0.2))
phi_wide = 0.5 * (1 - np.tanh((r_1d - R_1d) / 1.0))

ax3.plot(r_1d, phi0, "k-", linewidth=2, label=r"$\phi_0(r)$, width $=0.5$")
ax3.plot(r_1d, phi_narrow, "green", linestyle="--", linewidth=1.5, label="narrow (0.2)")
ax3.plot(r_1d, phi_wide, "green", linestyle="--", linewidth=1.5, label="wide (1.0)")
ax3.set_xlabel(r"Radial distance $r$")
ax3.set_ylabel(r"Density $\phi$")
ax3.set_title(
    "(iii) Profile Fluctuation\n(Irrelevant: Fast Diffusive Relaxation)",
    pad=10,
)
ax3.set_xlim(2, 8)
ax3.set_ylim(-0.1, 1.1)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right", fontsize=9)

# =============================================================================
# 整體標題與輸出
# =============================================================================
plt.suptitle(
    "Dimensionality Reduction in Active Nucleation: Surviving Fluctuations",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig("ambp_three_fluctuations.png", dpi=150, bbox_inches="tight")
plt.show()
