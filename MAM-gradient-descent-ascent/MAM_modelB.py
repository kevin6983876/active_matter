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

def lap(field):
    # 傳入的 field 形狀是 (Ncopy, L)
    # 1. 轉到頻域
    field_k = np.fft.fft(field, axis=-1)
    # 2. 乘上 -k^2 再轉回實域
    return np.fft.ifft(-k2 * field_k, axis=-1).real

# === 修改後的 Model B Hamiltonian ===
def Hamiltonian(h, rho, theta): 
    lap_rho = lap(rho)
    lap_theta = lap(theta)
    
    # 1. 決定性動力學部分: \nabla^2 (D \nabla^2 \rho + \rho - \rho^3)
    det_part = lap(-D * lap_rho - rho + rho**3)
    
    # 2. 守恆型雜訊項: -(a/2) * \theta * \nabla^2 \theta
    # 注意：Model B 的雜訊也必須是守恆的 (帶有 Laplacian)
    noise_part = -0.5 * aa * theta * lap_theta
    
    # 總和
    H = np.sum(det_part * theta + noise_part, axis=1)
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

def Lagrangian_Theo(h, rho, theta):
    lap_theta = lap(theta)
    # L = 0.5 * aa * \int |\nabla \theta|^2 dx = -0.5 * aa * \int \theta \nabla^2 \theta dx
    L_density = -0.5 * aa * theta * lap_theta
    return np.sum(L_density, axis=1)

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

Lx = 1.
L  = 2
h  = 1
Ncopy = 400 
Tmax = 2.


s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
aa = 2.   #noise amplitude


upward = True # choose if path from -1 to +1 (upward), or the opposite

D     = 0.001

solRho = [-1.0, 0.0, 1.0]

dtau = 0.02

iterations = 2000
plotStep   = 400

r = dtau/dnu

# === [新增] FFT 頻域波數矩陣 ===
# 產生對應長度 L 的 k 向量
k_vec = 2 * PI * np.fft.fftfreq(L, d=h)
k2 = k_vec**2  # 用來算 Laplacian (\nabla^2)
k4 = k_vec**4  # 用來算 Biharmonic (\nabla^4)

# [新增] IMEX 穩定化係數
gamma = 2.0  # 通常設為 2.0 左右即可完美壓制高頻爆炸

A_solve_upper_adapted = np.zeros((L,Ncopy,Ncopy))
B_solve_lower_adapted = np.zeros((L,Ncopy,Ncopy))

## Second order scheme (我們直接用二階，放棄一階以確保精度)
for kk in range(0,L):
    # [核心修改] 針對第 kk 個空間波數，計算它的穩定化強度
    stab = dtau * gamma * D * k4[kk]
    
    # 填寫 A 矩陣 (將 stab 藏進主對角線)
    np.fill_diagonal( A_solve_upper_adapted[kk], 1. + 3*r/2. + stab ) 
    np.fill_diagonal( A_solve_upper_adapted[kk][:-1, 1:], -2*r ) 
    np.fill_diagonal( A_solve_upper_adapted[kk][:-2, 2:], r/2. ) 
    A_solve_upper_adapted[kk][Ncopy-2,Ncopy-2] = 1 + r + stab
    A_solve_upper_adapted[kk][Ncopy-2,Ncopy-1] = -r
    A_solve_upper_adapted[kk][Ncopy-1,Ncopy-1] = 1 + stab

    # 填寫 B 矩陣 (將 stab 藏進主對角線)
    np.fill_diagonal( B_solve_lower_adapted[kk,1:Ncopy,1:Ncopy],  1. + 3*r/2. + stab ) 
    np.fill_diagonal( B_solve_lower_adapted[kk][1:, :-1], -2*r ) 
    np.fill_diagonal( B_solve_lower_adapted[kk][2:, :-2], r/2. ) 
    B_solve_lower_adapted[kk][1,1] = 1 + r + stab
    B_solve_lower_adapted[kk][1,0] = -r
    B_solve_lower_adapted[kk,0,0] = 1 + stab


########## ARRAY CREATION

rho   = np.zeros((Ncopy,L), dtype=complex)

theta = np.zeros((Ncopy,L), dtype=complex)

U = np.zeros((Ncopy,L), dtype=complex)
U_Fourier = np.zeros((Ncopy,L), dtype=complex)

V = np.zeros((Ncopy,L), dtype=complex)
V_Fourier = np.zeros((Ncopy,L), dtype=complex)

U2_Fourier = np.zeros((Ncopy,L), dtype=complex)
V2_Fourier = np.zeros((Ncopy,L), dtype=complex)

reaction_U = np.zeros((Ncopy,L), dtype=complex)
reaction_V = np.zeros((Ncopy,L), dtype=complex)


if(upward==True):
    rho1 = np.zeros(L, dtype=complex)
    rho2 = np.zeros(L, dtype=complex)
    
    # 起點：左高右低
    rho1[L//2] = solRho[2] 
    rho1[0] = solRho[0]
    rho1k = np.fft.fft(rho1)
    
    # 終點：左低右高
    rho2[L//2] = solRho[0]
    rho2[0] = solRho[2]
    rho2k = np.fft.fft(rho2)
else:
    rho1 = np.zeros(L, dtype=complex)
    rho2 = np.zeros(L, dtype=complex)
    
    # 起點：左低右高
    rho1[L//2] = solRho[0]
    rho1[0] = solRho[2]
    rho1k = np.fft.fft(rho1)
    
    # 終點：左高右低
    rho2[L//2] = solRho[2]
    rho2[0] = solRho[0]
    rho2k = np.fft.fft(rho2)


reaction_U_Fourier = np.zeros((Ncopy,L), dtype=complex)
reaction_V_Fourier = np.zeros((Ncopy,L), dtype=complex)


######  INITIAL CONDITIONS 


rho[0]       = rho1
rho[Ncopy-1] = rho2
# deterministically set the initial condition
amp = 0.4

# randomly set the initial condition
# noise_amp = 0.5  # noise amplitude
# np.random.seed(42) 

for j in range(1,Ncopy-1):
	tt = float(j)/Ncopy
	linear = rho1*(1-tt) + tt*rho2
	# deterministic initial condition
	bump = amp*np.square(np.sin(PI*np.arange(0,L)/L))*np.power(np.sin(PI*tt),2)
	# random initial condition
	# bump = noise_amp*np.random.normal(0, 1, size=L)*np.sin(PI*tt)
	bump = bump - np.mean(bump)
	rho[j] = linear + bump




U = rho + theta
V = rho - theta



###### Evolve complex array ######
for i in range(0, iterations+1):
	

	##################################################
	'''------------ FIRST STEP: update U ---------'''
	##################################################	
	#print 'UPDATE U...'
	#Compute nonlinear term in real space
	# dtau_rho = dt_theta + dH_drho
	# dtau_theta = dt_rho - dH_dtheta
	
	lap_rho = lap(rho)
	lap_theta = lap(theta)
	# dH_drho = D*(np.roll(theta,-1, axis=1) +  np.roll(theta,1, axis=1) -2* theta)/h**2 + theta - 3*rho**2 *theta + kappa*2*rho*np.outer(np.mean(theta, axis=1), np.ones(L))
	dH_drho = -D * lap(lap_theta) - (1 - 3*rho**2) * lap_theta
	
	# dH_dtheta = D*(np.roll(rho,-1, axis=1) +  np.roll(rho,1, axis=1) -2* rho)/h**2 + rho -rho**3 + kappa*np.outer(np.mean(rho**2,axis=1), np.ones(L)) + aa*theta
	dH_dtheta = lap(-D * lap_rho - rho + rho**3) - aa * lap_theta
	
	reaction_U = dH_drho - dH_dtheta
	
	#Compute Fourier transform of nonlinear terms
	reaction_U_Fourier[:] = np.fft.fft( reaction_U[:] )  #size Ncopy,L

	U_Fourier[:] = np.fft.fft( U[:]) # Ncopy * L
	V_Fourier[:] = np.fft.fft( V[:])

	
	#COMPUTE EVOLUTION for mode k, temporary array (Ncopy,L)
	
	temporaryArray = U_Fourier[:,:]

	temporaryArray[Ncopy-1] = (-V_Fourier[Ncopy-1,:] + 2*rho2k[:])#implement boundary conditions

	reaction_U_Fourier[Ncopy-1,:] = 0
	
	for kk in range(0,L):
		# U2_Fourier[:,kk] = solve_triangular( A_solve_upper_adapted[kk], temporaryArray[:,kk] + dtau*(reaction_U_Fourier[:,kk])) #size 2Ncopy
		# [新增] 為了物理平衡，在等式右邊補回顯式穩定項
		stab_term = gamma * D * k4[kk] * temporaryArray[:,kk]
		RHS = temporaryArray[:,kk] + dtau*(reaction_U_Fourier[:,kk] + stab_term)
		U2_Fourier[:,kk] = solve_triangular( A_solve_upper_adapted[kk], RHS)


	#Compute inverse Fourier transform of UV2
	U[:] = np.fft.ifft( U2_Fourier[:]).real #size 2Ncopy,L

	rho   = 0.5*(U+V)   #size Ncopy,L
	theta = 0.5*(U-V) #size Ncopy,L

	target_mass = np.mean(rho1) # 目標質量
	rho = rho - np.mean(rho, axis=1, keepdims=True) + target_mass
	theta = theta - np.mean(theta, axis=1, keepdims=True) # 鎖死 theta 漂移

	rho[0,:] = rho1
	rho[Ncopy-1,:] = rho2
	
	#print rho[0].real
	#print rho[Ncopy-1,:].real
	U = rho + theta #size Ncopy,L
	V = rho - theta #size Ncopy,L
	
	


	##################################################
	'''------------ SECOND STEP: update V ---------'''
	##################################################	
	#print 'UPDATE V...'
	#Compute nonlinear term in real space

	lap_rho = lap(rho)
	lap_theta = lap(theta)
	# dH_drho = D*(np.roll(theta,-1, axis=1) +  np.roll(theta,1, axis=1) -2* theta)/h**2 + theta - 3*rho**2 *theta + kappa*2*rho*np.outer(np.mean(theta, axis=1), np.ones(L))
	dH_drho = -D * lap(lap_theta) - (1 - 3*rho**2) * lap_theta
	
	# dH_dtheta = D*(np.roll(rho,-1, axis=1) +  np.roll(rho,1, axis=1) -2* rho)/h**2 + rho -rho**3 + kappa*np.outer(np.mean(rho**2,axis=1), np.ones(L)) + aa*theta
	dH_dtheta = lap(-D * lap_rho - rho + rho**3) - aa * lap_theta
	
	
	reaction_V = dH_drho + dH_dtheta


	#Compute Fourier transform of nonlinear terms
	reaction_V_Fourier[:] = np.fft.fft( reaction_V[:])
	
	U_Fourier[:] = np.fft.fft( U[:])
	V_Fourier[:] = np.fft.fft( V[:])


	
	temporaryArray = V_Fourier[:,:] 
	#implement boundary conditions
	temporaryArray[0] = (-U_Fourier[0,:] + 2*rho1k[:])
	reaction_V_Fourier[0,:] = 0
	
	
	#COMPUTE EVOLUTION for mode k
	
	for kk in range(0,L):
		# V2_Fourier[:,kk] = solve_triangular(B_solve_lower_adapted[kk], temporaryArray[:,kk] + dtau*(reaction_V_Fourier[:,kk]), lower=True)  #size 2Ncopy
		# [新增] 為了物理平衡，在等式右邊補回顯式穩定項
		stab_term = gamma * D * k4[kk] * temporaryArray[:,kk]
		RHS = temporaryArray[:,kk] + dtau*(reaction_V_Fourier[:,kk] + stab_term)
		V2_Fourier[:,kk] = solve_triangular(B_solve_lower_adapted[kk], RHS, lower=True)

	#Compute inverse Fourier transform of UV2
	V[:] = np.fft.ifft( V2_Fourier[:]).real #size 2Ncopy,L

	rho   = 0.5*(U+V)   #size Ncopy,L
	theta = 0.5*(U-V) #size Ncopy,L

	target_mass = np.mean(rho1) 
	rho = rho - np.mean(rho, axis=1, keepdims=True) + target_mass
	theta = theta - np.mean(theta, axis=1, keepdims=True)

	rho[0,:]       = rho1
	rho[Ncopy-1,:] = rho2

	
	U = rho + theta #size Ncopy,L
	V = rho - theta #size Ncopy,L
	



	##################################################
	'''------------ THIRD: PLOT DATA ---------'''
	##################################################	
	    
	if( i%plotStep == 0):
		
		plt.gcf()
		fig = plt.figure(figsize=(10,10),layout='constrained')
		# fig.tight_layout(pad=0.0)
		# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
		ax0 = fig.add_subplot(222)
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(223)
		ax3 = fig.add_subplot(224)
		Lag = Lagrangian(h, dnu, rho, theta).real
		actionS = dnu* np.sum(Lag)
		
		fig.suptitle(r'$N_\mathrm{copy}=$'+str(int(Ncopy))+r', $L=$'+str(L)+r', $\Delta \tau=$'+str("%.1e"%dtau)+r', $D=$'+str("%.1e"%D) +r', $T_\mathrm{max}=$'+str("%.1f"%Tmax)+r', $S=$'+str("%.6f"%(actionS))+', Time '+ str(i*dtau), fontsize=20)
		
		### VECTOR FIELD
		# Saction = dnu* np.sum(Lagrangian(h, dnu, rho, theta).real)
		# ax0.scatter(rho[:,0].real, rho[:,1].real, color='darkblue', s=2.2, zorder=15)
		# x = np.arange(solRho[0], solRho[2], 0.02)
		# y = np.arange(solRho[0], solRho[2], 0.02)

		# X, Y = np.meshgrid(x, y)
		# u = D*(2*Y-2*X)/h**2  + (X - X*X*X) + kappa*(X*X+Y*Y)/2.
		# v = D*(2*X-2*Y)/h**2  + (Y - Y*Y*Y) + kappa*(X*X+Y*Y)/2.

		# ax0.streamplot(x, y, u, v, density=1, color='grey')
		# ax0.set_xlabel(r'$\rho_1$', fontsize=20)
		# ax0.set_ylabel(r'$\rho_2$', fontsize=20)
		# ax0.set_xlim(solRho[0],solRho[2])
		# ax0.set_ylim(solRho[0],solRho[2])
		# ax0.tick_params(axis='both', which='major', labelsize=15)
		# ax0.set_aspect('equal')
		### [NEW] PLOT: Symmetry Breaking Projection (Mean vs Std)
		mean_rho = np.mean(rho.real, axis=1) 
		std_rho  = np.std(rho.real, axis=1)  	

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
		rho_map = rho.real.T  # Shape: (L, Ncopy)
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		x_edges = np.arange(L + 1) - 0.5 # Shape: (L+1,)
		im = ax1.pcolormesh(t_edges, x_edges, rho_map, 
                           cmap='bwr', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho$', fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_yticks([0,L-1])
		ax1.set_xlabel('Time', fontsize=20)
		ax1.set_ylabel('Space index (0 to L-1)', fontsize=20)
		Lag = Lagrangian(h, dnu, rho, theta).real
		Lag_theo = Lagrangian_Theo(h, rho, theta).real
		actionS = dnu* np.sum(Lag)

		### PLOT theta
		# ax2.set_aspect('equal')
		# ax2.plot((np.arange(0,Ncopy)/(float(Ncopy-1))), theta[:,0].real, linewidth=1, label=r'$\theta_1$' )
		# ax2.plot((np.arange(0,Ncopy)/(float(Ncopy-1))), theta[:,1].real, linewidth=1, label=r'$\theta_2$' )
		# ax2.legend(loc='best', fontsize=15)
		# ax2.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
		# ax2.set_ylabel(r'$\theta$', fontsize=20)
		# ax2.tick_params(axis='both', which='major', labelsize=15)
		# ax2.set_xlim(0,1)
		# ax2.set_ylim(-0.5,0.5)

		theta_map = theta.real.T  # Shape: (L, Ncopy)
		im2 = ax2.pcolormesh(t_edges, x_edges, theta_map, 
							cmap='bwr', 
							shading='flat',
							vmin=min(theta_map.min(), -0.5), vmax=max(theta_map.max(), 0.5))
		cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
		cbar2.set_label(r'$\theta$', fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=15)
		ax2.set_yticks([0,L-1])
		ax2.set_xlabel('Time', fontsize=20)
		ax2.set_ylabel('Space index (0 to L-1)', fontsize=20)

		### PLOT Lagrangian and Hamiltonian
		ax3.set_aspect('auto')
		ax3.plot(np.linspace(0,1,Ncopy), Lag, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax3.plot(np.linspace(0,1,Ncopy), Lag_theo, label=r'$L_{th}(\rho,\dot\rho)$', color='red'  ) 
		ax3.plot(np.linspace(0,1,Ncopy), Hamiltonian(h,rho, theta).real, label=r'$H(\rho,\theta)$', color='brown' ) 
		plt.legend(loc='best', fontsize=14)
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
		ax3.set_ylabel(r'$L$', fontsize=20)
		ax3.set_xlim(0,1)
		
		if(upward==True):
			plt.savefig('upward_L'+str(int(L))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
		else:
			plt.savefig('downward_L'+str(int(L))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'_D'+str(D)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
		plt.clf()
		plt.close()

# Animate for any L
animate_any_boxes(rho.real, theta.real, filename=f"boxes_L{int(L)}.mp4", dnu=dnu, skip=4, fps=25)












