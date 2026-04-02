#!/usr/bin/env python3
"""
Active Phase Separation：從 3D 密度場降維到 1D 半徑演化

1x2 子圖視覺化：
- 左：Eq. 5 角積分與擬設（形狀漲落 → 平均半徑 R_t）
- 右：Eq. 6 介面濾波器與連鎖律（密度變化集中在介面）
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# =============================================================================
# 畫布：左極座標、右直角
# =============================================================================
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="polar")
ax2 = fig.add_subplot(1, 2, 2)

theta = np.linspace(0, 2 * np.pi, 300)
R_t = 5.0

# =============================================================================
# 圖一：Eq. 5 角積分與擬設
# =============================================================================
# 多條帶隨機高頻雜訊的邊界 R(θ) = 5 + 0.3*noise(θ)，代表不同微狀態
np.random.seed(42)
n_realizations = 8
for _ in range(n_realizations):
    noise = np.random.randn(len(theta))
    n = len(noise)
    f = fft(noise)
    f[n//4:-n//4] = 0
    noise_smooth = np.real(ifft(f))
    noise_smooth = 0.3 * noise_smooth / (np.max(np.abs(noise_smooth)) + 1e-8)
    R_noisy = R_t + noise_smooth
    ax1.plot(theta, R_noisy, color="skyblue", linewidth=0.8, alpha=0.8)

# 完美平均半徑的正圓
ax1.plot(theta, np.ones_like(theta) * R_t, "k-", linewidth=2.5, label=r"$R_t = 5$ (mean)")
# 曲率修正等效半徑
ax1.plot(theta, np.ones_like(theta) * 4.8, "r--", linewidth=1.5, label=r"$R_t + \varphi_1/R_t$ (4.8)")
# 箭頭：從漲落邊界指向中心，代表角積分
theta_arrow = np.pi / 2
r_arrow_start = R_t + 0.4
ax1.annotate(
    r"$\int d\theta$ Angular Averaging",
    xy=(theta_arrow, 1.2),
    xytext=(theta_arrow, r_arrow_start),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    fontsize=10,
    ha="center",
)
ax1.set_title(
    "Eq. 5: Angular Integration & Ansatz\nSmashing 3D fluctuations into 1D $R_t$",
    pad=20,
)
ax1.set_ylim(0, 7)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.legend(loc="upper right", fontsize=9)

# =============================================================================
# 圖二：Eq. 6 介面濾波器與連鎖律
# =============================================================================
r = np.linspace(2, 8, 500)
R_t_1d = 5.0
# 曲線 A：φ(r) = 0.5*(1 - tanh(2*(r - 5)))
phi = 0.5 * (1 - np.tanh(2 * (r - R_t_1d)))
ax2.plot(r, phi, "b-", linewidth=2, label=r"$\varphi(r - R_t)$")

# 曲線 B：-φ'(r) = sech²(2*(r-5))，介面處尖峰
phi_prime = -0.5 * 2 * (1 / np.cosh(2 * (r - R_t_1d)) ** 2)
neg_phi_prime = -phi_prime
neg_phi_prime_plot = neg_phi_prime / neg_phi_prime.max() * 0.9
ax2.plot(r, neg_phi_prime_plot, "r-", linewidth=2, label=r"$-\varphi'(r - R_t)$ (Interface Probe)")
ax2.fill_between(r, 0, neg_phi_prime_plot, color="red", alpha=0.2)

# 註解：Eq. 6 LHS
ax2.annotate(
    "Eq. 6 LHS:\nDensity only changes HERE\nproportional to $\\dot{R}_t$",
    xy=(R_t_1d, 0.9),
    xytext=(3.2, 0.45),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9),
)
ax2.set_xlabel(r"Radial distance $r$")
ax2.set_ylabel(r"$\varphi$ / normalized $(-\varphi')$")
ax2.set_title(
    "Eq. 6: The Chain Rule Projection\nIsolating dynamics at the interface",
    pad=10,
)
ax2.set_xlim(2, 8)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

# =============================================================================
# 整體標題與輸出
# =============================================================================
plt.suptitle(
    "Mathematical Coarse-Graining: From Field Theory to Droplet Radius",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig("ambp_field_to_radius.png", dpi=150, bbox_inches="tight")
plt.show()
