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

from scipy.linalg import solve_triangular
#x = solve_triangular(a, b, lower=True)


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






################################
""" Start MAM """
##############################

# Lx = 1
# Ly = 50
# h  = 1
# Ncopy = 1600 
# Tmax = 40.


# s = np.linspace(0,Tmax,Ncopy)
# ds  = s[1]-s[0]
# dnu = s[1]-s[0]
# aa = 2.   #noise amplitude


# upward = False # choose if path from -1 to +1 (upward), or the opposite

# D     = 5



# dtau = 0.5

# r = dtau/dnu

resume_file = "checkpoints/checkpoint_b_2d_3.npz"
if os.path.exists(resume_file):
	data = np.load(resume_file)
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
else:
	end_iterations = 0
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
###### Evolve Loop (GDA Algorithm 1: path-time upwind + full reaction) ######
start_time = time.time()
print("end_iterations", end_iterations)
U = rho + theta
V = rho - theta
##################################################
'''------------ THIRD: PLOT DATA ---------'''
##################################################	
	
# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
plt.gcf()
rho_1d = rho.reshape(Ncopy, -1)
theta_1d = theta.reshape(Ncopy, -1)
fig = plt.figure(figsize=(10,10),layout='constrained')
# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
ax0 = fig.add_subplot(222)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
actionS = dnu* np.sum(Lag)*h**2

fig.suptitle(r'$N_\mathrm{copy}=$'+str(int(Ncopy))+r', box size $=('+str(Ly)+r'\times'+str(Lx)+r')$, h ='+str(h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +r', $T_\mathrm{max}=$'+str(Tmax)+"\n"+r', a ='+str(aa)+', Time '+ str(end_iterations*dtau), fontsize=20)

### [NEW] PLOT: Symmetry Breaking Projection (Mean vs Std)
mean_rho = np.mean(rho_1d.real, axis=1) 
std_rho  = np.std(rho_1d.real, axis=1)  	

im = ax0.scatter(std_rho, mean_rho, c=np.linspace(0,1,Ncopy), cmap='viridis', s=15, zorder=10) 
ax0.plot(std_rho, mean_rho, color='darkblue', linewidth=2, label='Instanton Path')
ax0.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Homogeneous (Symmetric)')

ax0.set_xlabel(r'Inhomogeneity $\sigma_\rho$ (Std Dev)', fontsize=15)
ax0.set_ylabel(r'Mean Density $\bar{\rho}$', fontsize=15)
ax0.set_title(r'Symmetry Breaking Projection', fontsize=16)

ax0.grid(True, linestyle=':', alpha=0.6)
ax0.set_xlim(left=-0.05, right=max(std_rho.max()*1.2, 0.5)) 
ax0.set_ylim(solRho[0]-0.2, solRho[2]+0.2)
ax0.legend(loc='best', fontsize=12)
## colorbar
cbar = fig.colorbar(im, ax=ax0, shrink=0.8)
cbar.set_label(r'Time', fontsize=20)

### PLOT rho
mid_y = Ly // 2
rho_slice = rho[:, mid_y, :].real.T
t_edges = np.linspace(0, Tmax, Ncopy + 1)
x_edges = np.arange(Lx + 1) - 0.5 # 現在 Y 軸是 X 空間座標
im = ax1.pcolormesh(t_edges, x_edges, rho_slice, 
                    cmap='coolwarm', 
                    shading='flat',
                    vmin=-1.5, vmax=1.5)
cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label(r'$\rho(x, y_{mid}, t)$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticks([0, Lx-1])
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel(f'X Coordinate (at y={mid_y})', fontsize=15)
Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
actionS = dnu* np.sum(Lag)*h**2

### PLOT theta
theta_slice = theta[:, mid_y, :].real.T
im2 = ax2.pcolormesh(t_edges, x_edges, theta_slice, 
                    cmap='bwr', 
                    shading='flat',
                    vmin=-0.5, vmax=0.5)
cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label(r'$\theta(x, y_{mid}, t)$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticks([0, Lx-1])
ax2.set_xlabel('Time', fontsize=20)
ax2.set_ylabel(f'X Coordinate (at y={mid_y})', fontsize=15)

### PLOT Lagrangian and Hamiltonian
ax3.set_aspect('auto')
ax3.plot(np.linspace(0,1,Ncopy), Lag*h**2, label=r'$L(\rho,\dot\rho)$'+r', $S=$'+str("%.6f"%(actionS)), color='black'  ) 
ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h**2, label=r'$H(\rho,\theta)$', color='brown' ) 
plt.legend(loc='best', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
ax3.set_ylabel(r'$L$', fontsize=20)
ax3.set_xlim(0,1)

if(upward==True):
	plt.savefig('upward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (end_iterations,)+'.png', format='png', bbox_inches='tight')
	end_time = time.time()
	print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
else:
	plt.savefig('downward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (end_iterations,)+'.png', format='png', bbox_inches='tight')
	end_time = time.time()
	print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
plt.clf()
plt.close()
def animate_2d_heatmap(rho, theta, filename="2d_evolution.mp4", dnu=None, skip=1, fps=30):
    """
    2D Heatmap Animation for both rho and theta
    rho, theta shape: (Ncopy, Ly, Lx)
    """
    rho = np.asarray(rho).real
    theta = np.asarray(theta).real
    Nt, Ly, Lx = rho.shape
    
    if dnu is None:
        dnu = 1.0 / max(Nt - 1, 1)

    # Downsample frames
    indices = np.arange(0, Nt, skip)
    if len(indices) > 300: # 限制最大幀數以免檔案太大
        indices = np.linspace(0, Nt - 1, 200, dtype=int)
    
    rho_sub = rho[indices]
    theta_sub = theta[indices]
    n_frames = len(indices)

    # Setup Figure (改成 1x2 的子圖，寬度加倍)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    
    # ---------------------------------------------------------
    # 1. 初始化 Rho 的圖 (左邊)
    # ---------------------------------------------------------
    im1 = ax1.imshow(rho_sub[0], cmap='bwr', vmin=-1.5, vmax=1.5, 
                     origin='lower', interpolation='nearest', animated=True)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cb1.set_label(r"Density $\rho$")
    ax1.set_title(r"Density $\rho$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # ---------------------------------------------------------
    # 2. 初始化 Theta 的圖 (右邊)
    # ---------------------------------------------------------
    # 自動抓取 theta 的最大絕對值，讓正負顏色對稱 (白色為 0)
    theta_max = np.abs(theta).max() 
    im2 = ax2.imshow(theta_sub[0], cmap='bwr', vmin=-theta_max, vmax=theta_max, 
                     origin='lower', interpolation='nearest', animated=True)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cb2.set_label(r"Noise Force $\theta$")
    ax2.set_title(r"Noise Force $\theta$")
    ax2.set_xlabel("x")
    # 共同的大標題
    fig.suptitle(f"Time: 0.00 | Path Evolution", fontsize=16)

    def update(frame_idx):
        # 同時更新兩張圖的資料
        im1.set_data(rho_sub[frame_idx])
        im2.set_data(theta_sub[frame_idx])
        
        # 更新大標題的時間
        t = indices[frame_idx] * dnu
        fig.suptitle(f"Time: {t:.2f} | Path Evolution")
        
        return im1, im2

    # 注意：這裡把 blit 改成 False，因為我們要更新 fig.suptitle
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000.0/fps)
    
    # Save
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "mp4":
        try:
            anim.save(filename, writer="ffmpeg", fps=fps, dpi=100)
        except:
            anim.save(filename.replace(".mp4", ".gif"), writer="pillow", fps=fps)
    else:
        anim.save(filename, writer="pillow", fps=fps)
        
    plt.close(fig)
    print(f"Saved 2D animation: {filename}")


# 傳入原始的 3D rho (Ncopy, Ly, Lx)
animate_2d_heatmap(rho.real, theta.real, filename=f"2d_sim_Lx{int(Lx)}_Ly{int(Ly)}.mp4", dnu=dnu, skip=2, fps=10)








