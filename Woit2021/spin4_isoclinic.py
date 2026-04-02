#!/usr/bin/env python3
"""
模組二：引力與弱力的雙人舞 —— Spin(4) 的拆解
視覺化選項：(1) 3D 流線總覽圖 — 一目了然比較不同 s
            (2) 2D 剖面圖 — 每個 s 單獨存檔

v_s = (-y, x, (2s-1)z)。3D 流線可直接看到「往 z=0 收斂」(s=0) vs 「從 z=0 發散」(s=1)。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import font_manager
from matplotlib import colors

# -----------------------------------------------------------------------------
# 字型
# -----------------------------------------------------------------------------
def _setup_cjk_or_english():
    cjk_candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Microsoft JhengHei", "SimHei",
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
if _use_chinese:
    FIG_TITLE_2D = r"模組二：Spin(4) 等角旋轉 — 2D 剖面流場 (不同 $s$)"
    FIG_TITLE_3D = r"模組二：Spin(4) 等角旋轉 — 3D 流線總覽（一目了然）"
    FIG_TITLE_XY_Z = r"模組二：xy 剖面（不同 $z$、不同 $s$）"
    FIG_TITLE_YZ_X = r"模組二：yz 剖面（不同 $x$、不同 $s$）"
    COL_XY, COL_XZ, COL_YZ = r"剖面 $z=0$ (xy)", r"剖面 $y=0$ (xz)", r"剖面 $x=0$ (yz)"
else:
    FIG_TITLE_2D = r"Module 2: Spin(4) isoclinic — 2D slice flow fields (varying $s$)"
    FIG_TITLE_3D = r"Module 2: Spin(4) isoclinic — 3D streamlines at a glance"
    FIG_TITLE_XY_Z = r"Module 2: xy slices at different $z$ and $s$"
    FIG_TITLE_YZ_X = r"Module 2: yz slices at different $x$ and $s$"
    COL_XY, COL_XZ, COL_YZ = r"Slice $z=0$ (xy)", r"Slice $y=0$ (xz)", r"Slice $x=0$ (yz)"

# -----------------------------------------------------------------------------
# 3D 流場 v_s = (-y, x, (2s-1)z) 與流線積分
# -----------------------------------------------------------------------------
def velocity_mix(x, y, z, s):
    """s=0 純右，s=1 純左。回傳 (vx, vy, vz)。"""
    vx = -y
    vy = x
    vz = (2*s - 1) * z
    return vx, vy, vz

def integrate_flow_3d(seed, s, n_steps=120, dt=0.025):
    """沿 v_s 積分得一條 3D 軌跡。"""
    traj = np.zeros((n_steps + 1, 3))
    traj[0] = seed
    for i in range(n_steps):
        x, y, z = traj[i]
        vx, vy, vz = velocity_mix(x, y, z, s)
        traj[i + 1] = traj[i] + dt * np.array([vx, vy, vz])
        if np.any(np.abs(traj[i + 1]) > 2.5):
            traj[i + 1] = traj[i]
    return traj

# 3D 流線的起點（在球面與赤道附近，方便看出扭絞）
stream_seeds = [
    (0.7, 0, 0.5), (0.5, 0.5, 0.3), (-0.4, 0.6, -0.2), (0, 0.75, 0.4),
    (-0.5, -0.4, 0.3), (0.6, -0.3, -0.4), (0.3, 0.6, 0.6), (-0.6, 0.2, -0.5),
]

# -----------------------------------------------------------------------------
# 2D 剖面流場（供對照）
# -----------------------------------------------------------------------------
L = 1.0
n_pts = 31

def flow_xy(x, y):
    return -y, x

def flow_xz(x, z, s, y_slice=0):
    """xz 平面（固定 y=y_slice）上的 in-plane 流速：(vx, vz) = (-y_slice, (2s-1)z)。"""
    return np.full_like(x, -y_slice), (2*s - 1) * z

def flow_yz(y, z, s, x_slice=0):
    """yz 平面（固定 x=x_slice）上的 in-plane 流速：(vy, vz) = (x_slice, (2s-1)z)。"""
    return np.full_like(y, x_slice), (2*s - 1) * z

s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
n_s = len(s_values)
n_cols = 3

# -----------------------------------------------------------------------------
# (1) 一目了然：一張圖五個 3D 子圖，同一視角、流線依 z 著色
# -----------------------------------------------------------------------------
def plot_3d_overview():
    fig = plt.figure(figsize=(16, 4))
    elev, azim = 22, 45
    r = 1.2

    for i, s in enumerate(s_values):
        ax = fig.add_subplot(1, n_s, i + 1, projection="3d")
        for seed in stream_seeds:
            traj = integrate_flow_3d(seed, s)
            z_vals = traj[:, 2]
            # 流線依 z 著色：低 z 偏藍、高 z 偏紅，一眼看出往哪收斂/發散
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="steelblue", alpha=0.85, lw=1.2)
            sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=z_vals, cmap="coolwarm",
                            s=4, alpha=0.7, vmin=-r, vmax=r)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("$y$", fontsize=9)
        ax.set_zlabel("$z$", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(r"$s={:.2f}$".format(s), fontsize=11)
    plt.colorbar(sc, ax=fig.get_axes(), shrink=0.6, label="$z$")
    fig.suptitle(FIG_TITLE_3D, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("spin4_isoclinic_3d_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: spin4_isoclinic_3d_overview.png")

# 流線用同一顏色時無法用整圖 colorbar，改為每條線用其 z 的顏色
def plot_3d_overview_colored():
    fig = plt.figure(figsize=(16, 4))
    elev, azim = 22, 45
    r = 1.2

    for i, s in enumerate(s_values):
        ax = fig.add_subplot(1, n_s, i + 1, projection="3d")
        for seed in stream_seeds:
            traj = integrate_flow_3d(seed, s)
            z_vals = traj[:, 2]
            # 每段線段依 z 著色：用 Line3DCollection 或逐段畫
            n_pt = len(traj)
            for k in range(n_pt - 1):
                ax.plot(traj[k:k+2, 0], traj[k:k+2, 1], traj[k:k+2, 2],
                        color=plt.cm.coolwarm(0.5 + 0.5 * (z_vals[k] + z_vals[k+1]) / (2 * r)), lw=1.5, alpha=0.9)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("$y$", fontsize=9)
        ax.set_zlabel("$z$", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(r"$s={:.2f}$".format(s), fontsize=11)
    fig.suptitle(FIG_TITLE_3D, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("spin4_isoclinic_3d_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: spin4_isoclinic_3d_overview.png")

# 簡化版：流線單色，靠標題與並排比較就夠直觀
def plot_3d_overview_simple():
    fig = plt.figure(figsize=(16, 4))
    elev, azim = 22, 45
    r = 1.2

    for i, s in enumerate(s_values):
        ax = fig.add_subplot(1, n_s, i + 1, projection="3d")
        for seed in stream_seeds:
            traj = integrate_flow_3d(seed, s)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="steelblue", alpha=0.85, lw=1.2)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel("$y$", fontsize=9)
        ax.set_zlabel("$z$", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(r"$s={:.2f}$".format(s), fontsize=11)
    fig.suptitle(FIG_TITLE_3D, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("spin4_isoclinic_3d_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: spin4_isoclinic_3d_overview.png")

# -----------------------------------------------------------------------------
# 繪圖：每個 s 一張 2D 圖（三欄 xy/xz/yz），可關閉
# -----------------------------------------------------------------------------
x_xy = np.linspace(-L, L, n_pts)
y_xy = np.linspace(-L, L, n_pts)
X_xy, Y_xy = np.meshgrid(x_xy, y_xy)
U_xy, V_xy = flow_xy(X_xy, Y_xy)

# xy 剖面：多個 z、多個 s（網格：列=z，欄=s）
# 說明：v_s = (-y, x, (2s-1)z)。平面內 (vx,vy)=(-y,x) 與 z、s 無關，所以每格「流線形狀」一樣。
# 但穿過該剖面的 v_z=(2s-1)z 隨 z 變，且三維總速率 |v|=sqrt(x^2+y^2+v_z^2) 也隨 z 變。
z_values_xy = [-0.8, -0.4, 0.0, 0.4, 0.8]

# x 剖面（yz 平面）：固定 x，流場 (vy, vz) = (x, (2s-1)z)
x_values_yz = [-0.8, -0.4, 0.0, 0.4, 0.8]
y_yz = np.linspace(-L, L, n_pts)
z_yz = np.linspace(-L, L, n_pts)
Y_yz_grid, Z_yz_grid = np.meshgrid(y_yz, z_yz)

def plot_xy_slices_multi_z():
    """每個 s 一張圖：該 s 下多個 z 的 xy 剖面。流場依總速率 |v| 上色，colorbar 範圍統一。"""
    n_z = len(z_values_xy)
    # 先算所有 (s, z) 剖面上的 |v| 範圍，統一 vmin, vmax
    speed_min, speed_max = np.inf, -np.inf
    for s in s_values:
        for z0 in z_values_xy:
            vz = (2*s - 1) * z0
            speed_2d = np.sqrt(X_xy**2 + Y_xy**2 + vz**2)
            speed_min = min(speed_min, speed_2d.min())
            speed_max = max(speed_max, speed_2d.max())
    norm = colors.Normalize(vmin=speed_min, vmax=speed_max)

    for s in s_values:
        fig, axes = plt.subplots(1, n_z, figsize=(2.5 * n_z, 2.8))
        if n_z == 1:
            axes = [axes]
        for j, z0 in enumerate(z_values_xy):
            ax = axes[j]
            vz = (2*s - 1) * z0
            speed_2d = np.sqrt(X_xy**2 + Y_xy**2 + vz**2)
            sp = ax.streamplot(
                X_xy, Y_xy, U_xy, V_xy,
                color=speed_2d,
                cmap="viridis",
                norm=norm,
                density=1.2,
                linewidth=0.8,
            )
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.set_aspect("equal")
            cbar = fig.colorbar(sp.lines, ax=ax, shrink=0.7, label=r"$|v|$", norm=norm)
            speed_at_origin = abs(vz)
            if _use_chinese:
                ax.set_title(
                    r"$z={:.2f}$" "\n"
                    r"$v_z={:.3f}$（穿過剖面）" "\n"
                    r"原點 $|v|={:.3f}$".format(z0, vz, speed_at_origin),
                    fontsize=9,
                )
            else:
                ax.set_title(
                    r"$z={:.2f}$" "\n"
                    r"$v_z={:.3f}$ (through slice)" "\n"
                    r"$|v|$ at origin $={:.3f}$".format(z0, vz, speed_at_origin),
                    fontsize=9,
                )
            ax.set_xlabel("$x$", fontsize=9)
            ax.set_ylabel("$y$", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            (FIG_TITLE_XY_Z + " — " + r"$s={:.2f}$"
            + ("\n（流線顏色 = 總速率 $|v|$，colorbar 範圍統一）" if _use_chinese else "\n(streamline color = total speed $|v|$, unified colorbar range)")).format(s),
            fontsize=11,
            y=1.02,
        )
        plt.tight_layout()
        outname = "spin4_isoclinic_xy_s{}.png".format(str(s).replace(".", "p"))
        plt.savefig(outname, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: {}".format(outname))

def plot_x_slices_multi_x():
    """每個 s 一張圖：該 s 下多個 x 的 yz 剖面（x 剖面）。流場依總速率 |v| 上色，colorbar 範圍統一。"""
    n_x = len(x_values_yz)
    # 統一 |v| 範圍：yz 剖面上 |v| = sqrt(vy^2 + vz^2) = sqrt(x0^2 + ((2s-1)*z)^2)
    speed_min, speed_max = np.inf, -np.inf
    for s in s_values:
        for x0 in x_values_yz:
            vz_term = (2*s - 1) * Z_yz_grid
            speed_2d = np.sqrt(x0**2 + vz_term**2)
            speed_min = min(speed_min, speed_2d.min())
            speed_max = max(speed_max, speed_2d.max())
    norm_x = colors.Normalize(vmin=speed_min, vmax=speed_max)

    for s in s_values:
        fig, axes = plt.subplots(1, n_x, figsize=(2.5 * n_x, 2.8))
        if n_x == 1:
            axes = [axes]
        for j, x0 in enumerate(x_values_yz):
            ax = axes[j]
            U_yz = np.full_like(Y_yz_grid, x0)
            V_yz = (2*s - 1) * Z_yz_grid
            speed_2d = np.sqrt(x0**2 + V_yz**2)
            sp = ax.streamplot(
                Y_yz_grid, Z_yz_grid, U_yz, V_yz,
                color=speed_2d,
                cmap="viridis",
                norm=norm_x,
                density=1.2,
                linewidth=0.8,
            )
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.set_aspect("equal")
            fig.colorbar(sp.lines, ax=ax, shrink=0.7, label=r"$|v|$", norm=norm_x)
            speed_at_origin = abs(x0)  # 原點 (y,z)=(0,0) 處 |v| = |vy| = |x0|
            if _use_chinese:
                ax.set_title(
                    r"$x={:.2f}$" "\n"
                    r"$v_y={:.3f}$（穿過剖面）" "\n"
                    r"原點 $|v|={:.3f}$".format(x0, x0, speed_at_origin),
                    fontsize=9,
                )
            else:
                ax.set_title(
                    r"$x={:.2f}$" "\n"
                    r"$v_y={:.3f}$ (through slice)" "\n"
                    r"$|v|$ at origin $={:.3f}$".format(x0, x0, speed_at_origin),
                    fontsize=9,
                )
            ax.set_xlabel("$y$", fontsize=9)
            ax.set_ylabel("$z$", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            (FIG_TITLE_YZ_X + " — " + r"$s={:.2f}$"
            + ("\n（流線顏色 = 總速率 $|v|$，colorbar 範圍統一）" if _use_chinese else "\n(streamline color = total speed $|v|$, unified colorbar range)")).format(s),
            fontsize=11,
            y=1.02,
        )
        plt.tight_layout()
        outname = "spin4_isoclinic_yz_s{}.png".format(str(s).replace(".", "p"))
        plt.savefig(outname, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: {}".format(outname))

USE_2D_SLICES = True  # 改 False 則只產出 3D 總覽

if USE_2D_SLICES:
    for s in s_values:
        fig, axes = plt.subplots(1, n_cols, figsize=(11, 4))
        ax = axes[0]
        ax.streamplot(X_xy, Y_xy, U_xy, V_xy, color="steelblue", density=1.2, linewidth=0.8)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(COL_XY)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        x_xz = np.linspace(-L, L, n_pts)
        z_xz = np.linspace(-L, L, n_pts)
        X_xz, Z_xz = np.meshgrid(x_xz, z_xz)
        U_xz, W_xz = flow_xz(X_xz, Z_xz, s)
        ax.streamplot(X_xz, Z_xz, U_xz, W_xz, color="steelblue", density=1.2, linewidth=0.8)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        ax.set_title(COL_XZ)
        ax.grid(True, alpha=0.3)
        ax = axes[2]
        y_yz = np.linspace(-L, L, n_pts)
        z_yz = np.linspace(-L, L, n_pts)
        Y_yz, Z_yz = np.meshgrid(y_yz, z_yz)
        V_yz, W_yz = flow_yz(Y_yz, Z_yz, s)
        ax.streamplot(Y_yz, Z_yz, V_yz, W_yz, color="steelblue", density=1.2, linewidth=0.8)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_xlabel("$y$")
        ax.set_ylabel("$z$")
        ax.set_title(COL_YZ)
        ax.grid(True, alpha=0.3)
        fig.suptitle(FIG_TITLE_2D + " — " + r"$s={:.2f}$".format(s), fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig("spin4_isoclinic_s{}.png".format(str(s).replace(".", "p")), dpi=150, bbox_inches="tight")
        plt.close()

# 產出 3D 總覽（一目了然）
plot_3d_overview_simple()

# 產出多 z 的 xy 剖面總覽（每個 s 一張，一圖多欄 = 不同 z）
plot_xy_slices_multi_z()

# 產出多 x 的 yz 剖面總覽（每個 s 一張，一圖多欄 = 不同 x）
plot_x_slices_multi_x()
