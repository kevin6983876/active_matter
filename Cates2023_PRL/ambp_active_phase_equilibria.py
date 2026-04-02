#!/usr/bin/env python3
"""
AMB+ 活性相平衡構造視覺化

並排對比：
- 左：被動系統對稱雙井自由能 f(φ) 與 Maxwell 共同切線（Binodal）
- 右：變數代換後的活性等效自由能 g(ψ) 與傾斜共同切線
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# =============================================================================
# 參數
# =============================================================================
alpha = -0.5   # 綜合活性參數，代表 (ζ - 2λ)/K
n_pts = 1000
phi = np.linspace(-2.0, 2.0, n_pts)

# =============================================================================
# 被動系統：雙井自由能 f(φ) 與化學勢 μ
# =============================================================================
f = -0.5 * phi**2 + 0.25 * phi**4
mu = -phi + phi**3   # df/dφ，與 np.gradient(f, phi) 一致

# 被動 Binodal：μ=0 的兩相為 φ₁=-1, φ₂=1，f(-1)=f(1)=-0.25
phi_binodal_1 = -1.0
phi_binodal_2 = 1.0
f_binodal = -0.25   # 共同切線為水平線 y = -0.25

# =============================================================================
# 變數代換 ψ(φ) 與等效自由能 g(ψ)
# =============================================================================
# 扭曲變數：ψ = (exp(αφ) - 1) / α
psi = (np.exp(alpha * phi) - 1) / alpha

# dg/dψ = μ ⇒ dg = μ dψ = μ (dψ/dφ) dφ = μ exp(αφ) dφ
# g(φ) = ∫ μ(φ) exp(αφ) dφ（數值積分）
integrand = mu * np.exp(alpha * phi)
if hasattr(integrate, "cumulative_trapezoid"):
    g = integrate.cumulative_trapezoid(integrand, phi, initial=0)
else:
    dphi = np.diff(phi)
    g_mid = (integrand[:-1] + integrand[1:]) / 2 * dphi
    g = np.concatenate([[0], np.cumsum(g_mid)])
# 平移 g 使最小值接近 0（可選，僅影響 Y 軸顯示）
g = g - g.min()

# 活性 Binodal：仍為同一化學勢的兩相 φ₁=-1, φ₂=1，對應到 ψ 空間
idx1 = np.argmin(np.abs(phi - phi_binodal_1))
idx2 = np.argmin(np.abs(phi - phi_binodal_2))
psi_1, psi_2 = psi[idx1], psi[idx2]
g_1, g_2 = g[idx1], g[idx2]
# 共同切線斜率
slope_tangent = (g_2 - g_1) / (psi_2 - psi_1) if psi_2 != psi_1 else 0
# 切線：g - g_1 = slope * (ψ - ψ_1)
psi_line = np.array([psi.min(), psi.max()])
g_tangent_line = g_1 + slope_tangent * (psi_line - psi_1)

# =============================================================================
# 視覺化
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ----- 左圖：被動 f(φ) -----
ax1.plot(phi, f, color="blue", linewidth=2, label=r"$f(\phi)$")
# 水平共同切線（Maxwell construction）
ax1.axhline(f_binodal, color="red", linestyle="--", linewidth=1.5, label="Maxwell Construction")
ax1.plot([phi_binodal_1, phi_binodal_2], [f_binodal, f_binodal], "ko", markersize=10, label="Binodal points")
ax1.set_xlabel(r"Order parameter $\phi$")
ax1.set_ylabel(r"Free energy $f(\phi)$")
ax1.set_title(r"Passive: Symmetric Free Energy $f(\phi)$")
ax1.set_xlim(-2, 2)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper center")

# ----- 右圖：活性 g(ψ) -----
ax2.plot(psi, g, color="orangered", linewidth=2, label=r"$g(\psi)$")
# 傾斜共同切線
ax2.plot(psi_line, g_tangent_line, color="red", linestyle="--", linewidth=1.5, label="Active Common Tangent")
ax2.plot([psi_1, psi_2], [g_1, g_2], "ko", markersize=10, label="Binodal points")
ax2.set_xlabel(r"Mapped variable $\psi$")
ax2.set_ylabel(r"Effective free energy $g(\psi)$")
ax2.set_title(r"Active: Mapped Potential $g(\psi)$")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper center")

# 變數變換公式（置於兩圖之間）
fig.text(
    0.5, 0.02,
    r"Variable mapping: $\psi =(\exp[\alpha\phi]-1)/\alpha$  (here $\alpha = (\zeta-2\lambda)/K$)",
    ha="center",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
)

plt.suptitle("Active Phase Equilibria: Passive vs Active Free Energy and Common Tangent", fontsize=12, y=1.02)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("ambp_active_phase_equilibria.png", dpi=150, bbox_inches="tight")
plt.show()
