from numpy import loadtxt
from matplotlib import cm
import math
import cmath
import time
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.linalg import solve_triangular, solve_banded

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)



import time
start_time = time.time()

PI = math.pi


##############################
""" DEFINE FUNCTIONS """
###############################
def build_rect_2d_laplacian(Nx, Ny, h):
	# ----------------------------------------
	# 1. 定義一個通用的 helper 函數來產生 1D 矩陣
	# ----------------------------------------
	def make_1d_diff_matrix(N):
		# 建立 1D 差分矩陣: 對角線 -2, 左右 1
		vals = [np.ones(N), -2*np.ones(N), np.ones(N)]
		offsets = [-1, 0, 1]
		D = sparse.diags(vals, offsets, shape=(N, N)).tolil()
		
		# 週期性邊界 (Periodic BC)
		D[0, N-1] = 1
		D[N-1, 0] = 1
		return sparse.csr_matrix(D)

	# ----------------------------------------
	# 2. 分別建立 X 和 Y 的 1D 算子
	# ----------------------------------------
	Dx = make_1d_diff_matrix(Nx) # 大小 Nx * Nx
	Dy = make_1d_diff_matrix(Ny) # 大小 Ny * Ny

	Ix = sparse.identity(Nx)     # 大小 Nx * Nx
	Iy = sparse.identity(Ny)     # 大小 Ny * Ny

	# ----------------------------------------
	# 3. 利用 Kronecker Product 組合 (核心改動)
	# ----------------------------------------
	# 注意順序：
	# row-major (C-style) flatten 時，Y 是快速變化的索引 (Fast index)，X 是慢速 (Slow index)
	# 所以 X 方向的微分算子要跟 Iy 做 tensor product (作用在大的 block 上)
	# Y 方向的微分算子要跟 Ix 做 tensor product (作用在每個 block 內部)

	# 擴散算子 = (d^2/dx^2) + (d^2/dy^2)
	if Nx == 1:
		# 僅 y 方向：Lap = d^2/dy^2 = Dy/h^2（與 1D 鏈 Ly 點相同）
		return Dy / (h**2)
	if Ny == 1:
		# 僅 x 方向：Lap = d^2/dx^2 = Dx/h^2
		return Dx / (h**2)
	else:
		Lap2D = sparse.kron(Dx, Iy) + sparse.kron(Ix, Dy)
		return Lap2D / (h**2)

from scipy.linalg import solve_triangular
#x = solve_triangular(a, b, lower=True)
def apply_lap_2d(field):
	# field (Ncopy, Ly, Lx) -> lap (Ncopy, Ly, Lx)
	N_current = field.shape[0]
	flat = field.reshape(N_current, -1)
	lap_flat = (Lap_Matrix.dot(flat.T)).T
	return lap_flat.reshape(N_current, Ly, Lx)

def Hamiltonian(h, rho, theta):
	# rho, theta: (Ncopy, Ly*Lx) 以配合 Lagrangian
	rho_3d = rho.reshape(Ncopy, Ly, Lx)
	theta_3d = theta.reshape(Ncopy, Ly, Lx)
	lap_rho = apply_lap_2d(rho_3d)
	lap_theta = apply_lap_2d(theta_3d)
	det_part = apply_lap_2d(-D * lap_rho - rho_3d + rho_3d**3)
	noise_part = -0.5 * aa * theta_3d * lap_theta
	H = np.sum(det_part * theta_3d + noise_part, axis=(1, 2))
	return H

def Lagrangian(h, ds, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	rhoDot      = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0))/(2*ds)
	rhoDot[0,:] = (np.roll(rho,-1,axis=0) - rho)[0,:] /(ds)
	rhoDot[Ncopy-1,:] = (-np.roll(rho,1,axis=0) + rho)[Ncopy-1,:]/(ds)
	Ham = Hamiltonian(h, rho, theta)
	L =  np.sum( rhoDot * theta, axis=1 ) - Ham
	return L


def animate_any_boxes(rho, theta, filename="boxes_evolution.mp4", dnu=None, skip=1, fps=30):
    """
    Generic Live bar chart animation for any L (L=2, 3, ...).
    """
    rho = np.asarray(rho).real
    theta = np.asarray(theta).real
    Nt, L = rho.shape  # 自動偵測 L
    
    if dnu is None:
        dnu = 1.0 / max(Nt - 1, 1)

    # Downsample
    indices = np.arange(0, Nt, skip)
    if len(indices) > 200:
        indices = np.linspace(0, Nt - 1, 150, dtype=int)
    rho_sub = rho[indices]
    theta_sub = theta[indices]
    n_frames = len(indices)

    # Setup Colors
    theta_max = max(np.abs(theta).max(), 0.01)
    norm = plt.Normalize(vmin=-theta_max, vmax=theta_max)
    cmap = plt.cm.coolwarm

    # Setup Figure
    fig, ax = plt.subplots(figsize=(max(4, L*1.5), 5), layout='constrained') # 寬度隨 L 自動調整
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-0.8, L - 0.2)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"Box {i+1}" for i in range(L)]) # 自動產生 Box 1, Box 2...
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$\rho$ (density)")
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(r"Noise Force ($\theta$)")

    # Initialize Bars
    initial_colors = [cmap(norm(theta_sub[0, i])) for i in range(L)]
    bars = ax.bar(range(L), rho_sub[0], color=initial_colors, width=0.6, edgecolor="k")
    
    ann = ax.text(0.5, 1.15, "", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")

    def update(frame_idx):
        rho_f = rho_sub[frame_idx]
        theta_f = theta_sub[frame_idx]
        t = indices[frame_idx] * dnu
        
        # Update each bar
        for i, b in enumerate(bars):
            b.set_height(rho_f[i])
            b.set_color(cmap(norm(theta_f[i])))
            
        ax.set_title(r"Time: {:.2f} | Action Density".format(t))
        
        # Generalized Symmetry Breaking Detection
        # 計算標準差，如果 > 0.1 代表有盒子長得不一樣
        std_dev = np.std(rho_f)
        if std_dev > 0.1:
            ann.set_text(f"Symmetry Breaking (std={std_dev:.2f})")
            ann.set_color("red")
        else:
            ann.set_text("Homogeneous")
            ann.set_color("black")
            
        return list(bars) + [ann]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000.0 / fps)
    
    # Save logic (same as before)
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "mp4":
        try:
            anim.save(filename, writer="ffmpeg", fps=fps, dpi=100)
        except Exception:
            anim.save(filename.replace(".mp4", ".gif"), writer="pillow", fps=fps, dpi=100)
    else:
        anim.save(filename, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved animation for L={L}: {filename}")



resume_files = ["checkpoints/checkpoint44.npz", "checkpoints/checkpoint43.npz", "checkpoints/checkpoint40.npz", "checkpoints/checkpoint41.npz","checkpoints/checkpoint42.npz"]
fig = plt.figure(figsize=(10,5),layout='constrained')
ax2 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)
colors = ['yellow', 'orange', 'salmon', 'red', 'purple']
for i, file in enumerate(resume_files):
    data = np.load(file)
    rho = data['rho']
    theta = data['theta']
    iteration = data['iteration']
    Lx = data['Lx']
    Ly = data['Ly']
    h = data['h']
    Ncopy = data['Ncopy']
    Tmax = data['Tmax']
    aa = data['aa']
    D = data['D']
    dtau = data['dtau']
    upward = data['upward']
    end_iterations = data['iterations']
    plotStep = data['plotStep']
    dnu = Tmax/Ncopy
    r = dtau/dnu
    solRho = np.array([-1.0, 0.0, 1.0]) # rho-, rhos, rho+ 
    print("end_iterations", end_iterations)
        # 1. 建立長方形的 2D Laplacian (大小是 (Lx*Ly) x (Lx*Ly))
    Lap_Matrix = build_rect_2d_laplacian(Lx, Ly, h)

    # 2. 建立隱式算子
    I_total = sparse.identity(Lx * Ly, format='csr')
    Operator_Matrix = I_total - dtau * D * Lap_Matrix

    # 3. 預先分解
    solve_2D = factorized(Operator_Matrix)

    U = rho + theta
    V = rho - theta
    rho_1d = rho.reshape(Ncopy, -1)
    theta_1d = theta.reshape(Ncopy, -1)
    plt.gcf()
    Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
    actionS = dnu* np.sum(Lag)*h*aa

    fig.suptitle(r'box size $=('+str(Ly)+r'\times'+str(Lx)+r')$, h ='+str(h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +r', upward ='+str(upward), fontsize=20)
    ### PLOT Lagrangian
    ax2.set_aspect('auto')
    ax2.plot(np.linspace(0,1,Ncopy), Lag*h*aa, label=r'$T_\mathrm{max}=$'+str(Tmax), color=colors[i] )  
    # ax2.legend(loc='best', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax2.set_ylabel(r'$L$', fontsize=20)
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.1,10.1)

    ### PLOT Hamiltonian
    ax3.set_aspect('auto')
    ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h*aa, label=r'$T_\mathrm{max}=$'+str(Tmax)+r', $S=$'+str("%.6f"%(actionS)), color=colors[i] ) 
    ax3.legend(loc='best', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax3.set_ylabel(r'$H$', fontsize=20)
    ax3.set_ylim(0,10.1)
    ax3.set_xlim(0,1)

plt.savefig('summary.png', format='png', bbox_inches='tight')
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
plt.clf()
plt.close()

# Animate for any L
# animate_any_boxes(rho_1d.real, theta_1d.real, filename=f"boxes_L{int(Ly)}.mp4", dnu=dnu, skip=4, fps=25)












