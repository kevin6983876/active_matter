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
from scipy import fftpack as sp_fft
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

def apply_lap_2d(field):
    # 完全捨棄有限差分矩陣，改用絕對精準的傅立葉頻域計算 Laplacian
    # field 的形狀為 (N_current, Ly, Lx)
    field_k = sp_fft.fft2(field, axes=(1, 2))
    
    # 乘上我們早就準備好的 2D 頻域 k^2 矩陣
    # k2_2d 的形狀是 (Ly, Lx)，Numpy 會自動漂亮地廣播 (Broadcast) 到所有 Ncopy 身上
    lap_k = -k2_2d * field_k
    
    # 轉回實數空間
    return sp_fft.ifft2(lap_k, axes=(1, 2)).real
def Hamiltonian(h, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	L_spatial = rho.shape[1]
	H = np.sum( (D*(np.roll(rho,-1, axis=1) +  np.roll(rho,1, axis=1) -2* rho)/h**2 + rho -rho**3 + kappa*np.outer(np.mean(rho**2,axis=1), np.ones(L_spatial)))*theta + 0.5*aa*theta**2     ,  axis=1)
	return H

def Lagrangian(h, ds, rho, theta): #return an array of size Ncopy, axis=1 sums the colums
	#rhoDot = (np.roll(rho,-1,axis=0) - rho)/ds
	#L = h*np.sum(np.power(theta.real,2),axis=1)
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


resume_files = ["checkpoints/checkpoint6.npz", "checkpoints/checkpoint6_1.npz", "checkpoints/checkpoint6_3.npz", "checkpoints/checkpoint6_4.npz"]
fig = plt.figure(figsize=(10,5),layout='constrained')
ax2 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)
colors = ['yellow', 'orange', 'red', 'purple']
for i, file in enumerate(resume_files):
    data = np.load(file)
    rho = data['rho']
    theta = data['theta']
    iteration = data['iteration']
    Lx = int(data['Lx'])
    Ly = int(data['Ly'])
    h = data['h']
    Ncopy = data['Ncopy']
    Tmax = data['Tmax']
    aa = data['aa']
    D = data['D']
    kappa = data['kappa']
    dtau = data['dtau']
    upward = data['upward']
    end_iterations = data['iterations']
    plotStep = data['plotStep']
    # === IMEX 穩定化（對應 modified 的 k4、gamma）===
    gamma = 0.5   # 與 modified 相同，壓制高頻
    # 2D Fourier 波數（與 fft2(..., axes=(1,2)) 對應）
    kx = 2 * PI * sp_fft.fftfreq(Lx, d=h)
    ky = 2 * PI * sp_fft.fftfreq(Ly, d=h)
    # 若 Lx 或 Ly 為 1，確保為 1D 且長度正確
    if np.isscalar(kx) or kx.size == 1:
        kx = np.atleast_1d(kx)
    if np.isscalar(ky) or ky.size == 1:
        ky = np.atleast_1d(ky)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')   # shape (Ly, Lx)
    k2_2d = KX**2 + KY**2
    k4_2d = k2_2d**2
    k4_flat = k4_2d.ravel()   # shape (Ly*Lx,) 與 U_Fourier.reshape(Ncopy,-1) 的 column 對應
    k2_flat = k2_2d.ravel()   # [新增] 準備 k2 的 flat 陣列
    dnu = Tmax/Ncopy
    r = dtau/dnu
    solRho = np.sort(np.roots([-1, kappa, 1, 0])) # rho-, rhos, rho+ 
    print("end_iterations", end_iterations)
    U = rho + theta
    V = rho - theta
    rho_1d = rho.reshape(Ncopy, -1)
    theta_1d = theta.reshape(Ncopy, -1)
    plt.gcf()
    Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
    actionS = dnu* np.sum(Lag)*h*aa

    fig.suptitle(r'$N_t=$'+str(Ncopy)+r', $N_x*h=$'+str(Ly*h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +r', $\kappa=$'+str(kappa), fontsize=20)
    ### PLOT Lagrangian
    ax2.set_aspect('auto')
    ax2.plot(np.linspace(0,1,Ncopy), Lag*h*aa, label=r'$T_\mathrm{max}=$'+str(Tmax), color=colors[i] )  
    

    # ax2.legend(loc='best', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax2.set_ylabel(r'$L$', fontsize=20)
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.1,max(Lag.max()*h*aa, Hamiltonian(h,rho_1d, theta_1d).real.max()*h*aa)*1.2)

    ### PLOT Hamiltonian
    ax3.set_aspect('auto')
    ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real*h*aa, label="$T_\mathrm{max}=$" +str(Tmax)+r', $S=$'+str("%.6f"%(actionS)), color=colors[i] ) 
    ax3.legend(loc='best', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax3.set_ylabel(r'$H$', fontsize=20)
    ax3.set_ylim(-0.01,max(Hamiltonian(h,rho_1d, theta_1d).real.max()*h*aa, Lag.max()*h*aa)*1.2)
    ax3.set_xlim(0,1)

plt.savefig('summary.png', format='png', bbox_inches='tight')
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
plt.clf()
plt.close()

# Animate for any L
# animate_any_boxes(rho_1d.real, theta_1d.real, filename=f"boxes_L{int(Ly)}.mp4", dnu=dnu, skip=4, fps=25)












