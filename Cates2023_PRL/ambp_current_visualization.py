#!/usr/bin/env python3
"""
AMB+ (Active Model B+) 微觀密度場與活性機率通量視覺化

視覺化液滴密度場 phi 以及由 zeta (ζ) 與 lambda (λ) 驅動的活性通量 J。
使用者可調整 zeta 與 lambda_param 後重新執行，觀察通量型態變化。
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 活性參數（可調整以觀察通量型態變化）
# =============================================================================
zeta = 2.0        # ζ：切向渦流強度 (tangential flow)
lambda_param = 2.0  # λ：法向推擠強度 (normal flow)

# =============================================================================
# 空間網格與密度場
# =============================================================================
n = 128
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
X, Y = np.meshgrid(x, y)

# 液滴半徑（可視需求調整）
R = 5.0
# 密度場：tanh 介面 + 微小高斯雜訊
np.random.seed(42)
phi = np.tanh(R - np.sqrt(X**2 + Y**2)) + 0.02 * np.random.randn(n, n)

# 梯度（介面法向量方向）
# np.gradient(phi)[0] = d(phi)/dy, [1] = d(phi)/dx
grad_y, grad_x = np.gradient(phi, y, x)
grad_sq = grad_x**2 + grad_y**2  # 將效應集中在介面附近

# =============================================================================
# 活性機率通量 J（AMB+）
# =============================================================================
# 切向渦流（由 ζ 驅動）
J_tangential_x = -grad_y * grad_sq * zeta
J_tangential_y = grad_x * grad_sq * zeta

# 法向推擠（由 λ 驅動）
J_normal_x = grad_x * grad_sq * lambda_param
J_normal_y = grad_y * grad_sq * lambda_param

# 總通量
J_x = J_tangential_x + J_normal_x
J_y = J_tangential_y + J_normal_y

# 通量大小（用於流線著色）
J_mag = np.sqrt(J_x**2 + J_y**2)

# =============================================================================
# 視覺化
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1)

# 底圖：密度場 phi（coolwarm：紅=液相，藍=氣相）
im = ax.contourf(X, Y, phi, levels=40, cmap="coolwarm")
plt.colorbar(im, ax=ax, label=r"$\phi$ (density)")
ax.contour(X, Y, phi, levels=[0], colors="k", linewidths=0.5, alpha=0.5)

# 流線圖：活性通量 (J_x, J_y)，顏色對應 |J|
strm = ax.streamplot(
    X[0], Y[:, 0], J_x, J_y,
    color=J_mag,
    cmap="viridis",
    linewidth=1.2,
    density=1.2,
    arrowsize=1.2,
    minlength=0.1,
)
cbar_strm = fig.colorbar(strm.lines, ax=ax)
cbar_strm.set_label(r"$|\mathbf{J}|$", fontsize=11)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title(
    f"AMB+ Microscopic Current | $\\zeta$={zeta}, $\\lambda$={lambda_param}"
)
plt.tight_layout()
plt.savefig("ambp_current.png", dpi=150, bbox_inches="tight")
