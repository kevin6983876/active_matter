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

def normL2(h, array): #norm L2, spacing h for integration
	norm = math.sqrt( h * np.sum( np.square( array.real ) ) )
	return norm

def normL2Big(h, array): #norm L2, spacing h for integration
	norm = np.sqrt( h * np.sum( np.square( array.real ), axis=1 ) )
	return norm




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
# kappa = 0.0



# dtau = 0.5

# r = dtau/dnu
# print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'D =', D, 'kappa =', kappa, 'dtau =', dtau, 'iterations =', iterations, 'Tmax =', Tmax)

resume_file = "checkpoints/checkpoint22.npz"
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
	kappa = data['kappa']
	dtau = data['dtau']
	upward = data['upward']
	end_iterations = data['iterations']
	plotStep = data['plotStep']
else:
	end_iterations = 0
dnu = Tmax/Ncopy
r = dtau/dnu
solRho = np.sort(np.roots([-1, kappa, 1, 0])) # rho-, rhos, rho+ 
###### Evolve Loop (GDA Algorithm 1: path-time upwind + full reaction) ######
# Apply 2D Laplacian to each path slice: (Ncopy, Ly*Lx)
def apply_lap_2d(field):
	# field (Ncopy, Ly, Lx) -> lap (Ncopy, Ly, Lx)
	flat = field.reshape(Ncopy, -1)
	lap_flat = (Lap_Matrix.dot(flat.T)).T
	return lap_flat.reshape(Ncopy, Ly, Lx)
start_time = time.time()
print("end_iterations", end_iterations)
U = rho + theta
V = rho - theta
##################################################
'''------------ THIRD: PLOT DATA ---------'''
##################################################	
	
# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
rho_1d = rho.reshape(Ncopy, -1)
theta_1d = theta.reshape(Ncopy, -1)
plt.gcf()
fig = plt.figure(figsize=(10,10),layout='constrained')
# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
ax0 = fig.add_subplot(222)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
actionS = dnu* np.sum(Lag)

fig.suptitle(r'$N_\mathrm{copy}=$'+str(int(Ncopy))+r', box size $=('+str(Ly)+r'\times'+str(Lx)+r')$, h ='+str(h)+r', $\Delta \tau=$'+str(dtau)+r', $D=$'+str(D) +r', $T_\mathrm{max}=$'+str(Tmax)+"\n"+str("kappa =")+str(kappa)+r', a ='+str(aa)+', Time '+ str(end_iterations*dtau), fontsize=20)

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
rho_map = rho_1d.real.T  # Shape: (Ly*Lx, Ncopy)
t_edges = np.linspace(0, Tmax, Ncopy + 1)
n_space = rho_1d.shape[1]  # Ly*Lx
x_edges = np.arange(n_space + 1) - 0.5
im = ax1.pcolormesh(t_edges, x_edges, rho_map, 
					cmap='coolwarm', 
					shading='flat',
					vmin=-1.5, vmax=1.5)
cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label(r'$\rho$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticks([0, n_space-1])
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel('Space index (0 to L-1)', fontsize=20)
Lag = Lagrangian(h, dnu, rho_1d, theta_1d).real
actionS = dnu* np.sum(Lag)

### PLOT theta
theta_map = theta_1d.real.T  # Shape: (Ly*Lx, Ncopy)
im2 = ax2.pcolormesh(t_edges, x_edges, theta_map, 
					cmap='bwr', 
					shading='flat',
					vmin=-0.5, vmax=0.5)
cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label(r'$\theta$', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticks([0, n_space-1])
ax2.set_xlabel('Time', fontsize=20)
ax2.set_ylabel('Space index (0 to L-1)', fontsize=20)

### PLOT Lagrangian and Hamiltonian
ax3.set_aspect('auto')
ax3.plot(np.linspace(0,1,Ncopy), Lag, label=r'$L(\rho,\dot\rho)$'+r', $S=$'+str("%.6f"%(actionS)), color='black'  ) 
ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho_1d, theta_1d).real, label=r'$H(\rho,\theta)$', color='brown' ) 
plt.legend(loc='best', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
ax3.set_ylabel(r'$L$', fontsize=20)
ax3.set_xlim(0,1)

if(upward==True):
	plt.savefig('upward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_kappa'+str(kappa)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (end_iterations,)+'.png', format='png', bbox_inches='tight')
	end_time = time.time()
	print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
else:
	plt.savefig('downward_L'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_kappa'+str(kappa)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (end_iterations,)+'.png', format='png', bbox_inches='tight')
	end_time = time.time()
	print(f"Time taken: {end_time - start_time} seconds", "iteration", end_iterations)
plt.clf()
plt.close()
# Animate for any L
# animate_any_boxes(rho_1d.real, theta_1d.real, filename=f"boxes_L{int(Ly)}.mp4", dnu=dnu, skip=4, fps=25)












