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


resume_files = ["checkpoints/checkpoint16.npz", "checkpoints/checkpoint17.npz", "checkpoints/checkpoint24.npz", "checkpoints/checkpoint25.npz"]
fig = plt.figure(figsize=(10,5),layout='constrained')
ax2 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)
colors = ['yellow', 'orange', 'red', 'purple']
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
    kappa = data['kappa']
    dtau = data['dtau']
    upward = data['upward']
    end_iterations = data['iterations']
    plotStep = data['plotStep']
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
    actionS = dnu* np.sum(Lag)*h

    fig.suptitle(r'$N_\mathrm{copy}=$'+'varied'+r', box size $=('+str(Ly)+r'\times'+str(Lx)+r')$, h ='+str(h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +"\n"+str("kappa =")+str(kappa)+r', a ='+str(aa), fontsize=20)
    ### PLOT Lagrangian
    ax2.set_aspect('auto')
    ax2.plot(np.linspace(0,1,Ncopy), Lag, label=r'$T_\mathrm{max}=$'+str(Tmax), color=colors[i] )  
    # ax2.legend(loc='best', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax2.set_ylabel(r'$L$', fontsize=20)
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.1,3.1)

    ### PLOT Hamiltonian
    ax3.set_aspect('auto')
    ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real, label=r'$T_\mathrm{max}=$'+str(Tmax)+r', $S=$'+str("%.6f"%(actionS)), color=colors[i] ) 
    ax3.legend(loc='best', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
    ax3.set_ylabel(r'$H$', fontsize=20)
    ax3.set_ylim(0,0.5)
    ax3.set_xlim(0,1)

plt.savefig('summary.png', format='png', bbox_inches='tight')
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
plt.clf()
plt.close()

# Animate for any L
# animate_any_boxes(rho_1d.real, theta_1d.real, filename=f"boxes_L{int(Ly)}.mp4", dnu=dnu, skip=4, fps=25)












