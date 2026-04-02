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
from scipy import fft as sp_fft
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



############# 2D SPARSE SOLVER SETUP ##########
def apply_lap_2d(field):
    # 完全捨棄有限差分矩陣，改用絕對精準的傅立葉頻域計算 Laplacian
    # field 的形狀為 (N_current, Ly, Lx)
    field_k = sp_fft.fft2(field, axes=(1, 2))
    
    # 乘上我們早就準備好的 2D 頻域 k^2 矩陣
    # k2_2d 的形狀是 (Ly, Lx)，Numpy 會自動漂亮地廣播 (Broadcast) 到所有 Ncopy 身上
    lap_k = -k2_2d * field_k
    
    # 轉回實數空間
    return sp_fft.ifft2(lap_k, axes=(1, 2)).real

def apply_grad_y(field):
    """
    ∂_y field (1D along y), via FFT.
    field shape: (N_current, Ly, Lx)
    returns: same shape (real part)
    """
    field_k = sp_fft.fft2(field, axes=(1, 2))
    grad_k = 1j * KY * field_k          # ∂_y ↔ i*k_y
    return sp_fft.ifft2(grad_k, axes=(1, 2)).real

def Hamiltonian_KH_1D(Pe, rho, m, p_rho, p_m):
    """
    KH Hamiltonian (1D along y).
    Inputs are flattened in space: (Ncopy, Ly*Lx) like your current Hamiltonian().
    Returns H per path slice: shape (Ncopy,)
    """
    rho3 = rho.reshape(Ncopy, Ly, Lx)
    m3   = m.reshape(Ncopy, Ly, Lx)
    pr3  = p_rho.reshape(Ncopy, Ly, Lx)
    pm3  = p_m.reshape(Ncopy, Ly, Lx)

    drho = apply_grad_y(rho3)
    dm   = apply_grad_y(m3)
    gpr  = apply_grad_y(pr3)
    gpm  = apply_grad_y(pm3)

    C11 = rho3 * (1.0 - rho3)
    C12 = m3   * (1.0 - rho3)
    C22 = C11

    # Drift terms
    term1 = gpr * (Pe * m3 * (1.0 - rho3) - drho)
    term2 = gpm * (Pe * rho3 * (1.0 - rho3) - dm)
    term_react = -2.0 * pm3 * m3

    # Conserved-noise quadratic form: (∂p)^T C (∂p) in 1D
    term_quad = C11 * (gpr**2) + 2.0 * C12 * (gpr * gpm) + C22 * (gpm**2)

    # Nonconserved piece
    term_noncons = rho3 * (pm3**2)

    H_density = term1 + term2 + term_react + term_quad + term_noncons
    return np.sum(H_density, axis=(1, 2))


def Lagrangian_KH_1D(Pe, ds, rho, m, p_rho, p_m):
    """
    L = sum( rhoDot*p_rho + mDot*p_m ) - H
    All inputs: (Ncopy, Ly*Lx)
    Returns L per slice: shape (Ncopy,)
    """
    rhoDot = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2*ds)
    mDot   = (np.roll(m,   -1, axis=0) - np.roll(m,   1, axis=0)) / (2*ds)

    rhoDot[0, :]     = (np.roll(rho, -1, axis=0) - rho)[0, :] / ds
    rhoDot[Ncopy-1,:]= (-np.roll(rho, 1, axis=0) + rho)[Ncopy-1, :] / ds
    mDot[0, :]       = (np.roll(m, -1, axis=0) - m)[0, :] / ds
    mDot[Ncopy-1,:]  = (-np.roll(m, 1, axis=0) + m)[Ncopy-1, :] / ds

    Ham = Hamiltonian_KH_1D(Pe, rho, m, p_rho, p_m)
    kin = np.sum(rhoDot * p_rho + mDot * p_m, axis=1)
    return kin - Ham


################################
""" Start MAM """
##############################

Lx = 1
Ly = 400
h  = 0.025
Ncopy = 400
Tmax = 20.

s = np.linspace(0,Tmax,Ncopy)
ds  = s[1]-s[0]
dnu = s[1]-s[0]
# aa = 2.   #noise amplitude
Pe = 6.0   # TODO: pick your Péclet-like parameter (must match KH paper's convention)


upward = True # choose if path from -1 to +1 (upward), or the opposite
rho_0 = 0.55
dtau = 0.01 

iterations = 9100
plotStep   = 100

# theoreticla spinodal points
high =3/4+1/4*np.sqrt(1-16/Pe**2)
low =3/4-1/4*np.sqrt(1-16/Pe**2)
print('theoretical spinodal points: high =', high, 'low =', low)
r = dtau/dnu
resume_file = "KHcheckpoints/checkpoint1.npz"
relaxed_file = 'KHcheckpoints/relaxed.npz'
print('conditions: Ly,Lx,Ncopy =', Ly,Lx,Ncopy, 'h =', h, 'Pe =', Pe, 'dtau =', dtau, 'iterations =', iterations, 'Tmax =', Tmax)
# === IMEX 穩定化（對應 modified 的 k4、gamma）===
gamma = 2.0   # 與 modified 相同，壓制高頻
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

############# PATH-TIME UPWIND MATRICES（per-mode + IMEX）##########
A_banded = np.zeros((Ly, 3, Ncopy))
B_banded = np.zeros((Ly, 3, Ncopy))

for col in range(Ly):
	stab = dtau * gamma * (2.0 * k2_flat[col])
	# A: second order + stab on diagonal
	A_banded[col, 2, :] = 1. + 3*r/2. + stab
	A_banded[col, 2, Ncopy-2] = 1. + r + stab
	A_banded[col, 2, Ncopy-1] = 1. #+ stab

	# 2. First upper diagonal
	A_banded[col, 1, 1:] = -2*r
	A_banded[col, 1, Ncopy-1] = -r

	# 3. Second upper diagonal
	A_banded[col, 0, 2:] = r/2.

	# ================= B 矩陣 (Lower Triangular) =================
	# scipy 規定下三角帶狀矩陣：row 0 是主對角線，row 1 是一階下方，row 2 是二階下方

	# 1. Main diagonal
	B_banded[col, 0, :] = 1. + 3*r/2. + stab
	B_banded[col, 0, 1] = 1. + r + stab
	B_banded[col, 0, 0] = 1.0 #+ stab 

	# 2. First lower diagonal
	B_banded[col, 1, :-1] = -2*r
	B_banded[col, 1, 0] = -r

	# 3. Second lower diagonal
	B_banded[col, 2, :-2] = r/2.


########## ARRAY CREATION

rho   = np.zeros((Ncopy, Ly, Lx), dtype=complex)
m     = np.zeros((Ncopy, Ly, Lx), dtype=complex)
p_rho = np.zeros((Ncopy, Ly, Lx), dtype=complex)
p_m   = np.zeros((Ncopy, Ly, Lx), dtype=complex)

# 兩套 U/V（每個 (q,p) 一套）
U_rho = rho + p_rho
V_rho = rho - p_rho
U_m   = m + p_m
V_m   = m - p_m

U_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
U_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
V_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

reaction_U_rho = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_rho = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_U_m   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_m   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

reaction_U_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_rho_Fourier = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_U_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)
reaction_V_m_Fourier   = np.zeros((Ncopy,Ly,Lx), dtype=complex)

y_coords = np.arange(Ly)
x_coords = np.arange(Lx)
Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
if(upward==True):
	W = Ly/2
	interface_thickness = 12  # 介面的平滑度 (建議 2~3，讓 FFT 算導數時不會爆炸)
	profile = 0.5*(np.tanh((Y - (Ly/2.0 - W/2.0)) / interface_thickness) - np.tanh((Y - (Ly/2.0 + W/2.0)) / interface_thickness))
	rho1 = profile + 0j
	if rho_0 > 0.5:
		rho1 = (rho1*(1-rho_0)*2+2*rho_0-1)/np.mean(rho1*(1-rho_0)*2+2*rho_0-1)*rho_0
	else:
		rho1 = rho1/np.mean(rho1)*rho_0
	rho1k = np.fft.fft2(rho1)
	grad_rho1 = sp_fft.ifft2(1j * KY * rho1k).real
	m1 = np.zeros((Ly,Lx), dtype=complex)
	m1_real = (1.0 / Pe) * grad_rho1 / (1.0 - rho1.real)
	m1 = m1_real + 0j
	
	W = Ly/2
	interface_thickness = 200  # 介面的平滑度 (建議 2~3，讓 FFT 算導數時不會爆炸)
	profile = 0.5*(np.tanh((Y - (Ly/2.0 - W/2.0)) / interface_thickness) - np.tanh((Y - (Ly/2.0 + W/2.0)) / interface_thickness))
	rho2 = profile + 0j
	if rho_0 > 0.5:
		rho2 = (rho2*(1-rho_0)*2+2*rho_0-1)/np.mean(rho2*(1-rho_0)*2+2*rho_0-1)*rho_0
	else:
		rho2 = rho2/np.mean(rho2)*rho_0
	rho2k = np.fft.fft2(rho2)
	grad_rho2 = sp_fft.ifft2(1j * KY * rho2k).real
	m2 = np.zeros((Ly,Lx), dtype=complex)
	# m2_real = (1.0 / Pe) * grad_rho2 / (1.0 - rho2.real)
	# m2 = m2_real + 0j

# check if mass is conserved
print("mass of rho1", np.sum(rho1))
print("mass of rho2", np.sum(rho2))
# ==========================================
# [新增] 預先放鬆 (Pre-relaxation) 區塊
# 目的：讓銳利的初始猜測自然擴散成真實的物理穩態
# ==========================================
print("Relaxing Initial and Final states to exact Model B equilibria...")
dt_relax = 0.0001  # 對於你設定的 h=0.3, D=0.002，這個步長非常安全
relax_steps = 20000	

if os.path.exists(relaxed_file):
	data = np.load(relaxed_file)
	rho1 = data['rho1']
	rho2 = data['rho2']
	m1 = data['m1']
	m2 = data['m2']
else:	
	def relax_kh_physical_states(rho_state, m_state, tol=1e-7, max_steps=500000):
		rho_2d = rho_state.copy().real
		m_2d   = m_state.copy().real
		target_mass = np.mean(rho_2d)
		print("開始尋找 KH2018 物理穩態 (dot(rho)=dot(m)=0)...")
		for step in range(1, max_steps + 1):
			rho_old = rho_2d.copy()
			m_old   = m_2d.copy()
			# FFT components (reuse for Laplacian)
			rho_k = sp_fft.fft2(rho_2d)
			m_k   = sp_fft.fft2(m_2d)
			# ∂_y^2 f  (1.5D: Lx=1 => k2_2d effectively KY^2)
			rho_dd = sp_fft.ifft2(-k2_2d * rho_k).real
			m_dd   = sp_fft.ifft2(-k2_2d * m_k).real
			# ∂_y [ m(1-rho) ] and ∂_y [ rho(1-rho) ]
			fm = m_2d * (1.0 - rho_2d)
			fr = rho_2d * (1.0 - rho_2d)
			grad_fm = sp_fft.ifft2(1j * KY * sp_fft.fft2(fm)).real
			grad_fr = sp_fft.ifft2(1j * KY * sp_fft.fft2(fr)).real
			# KH2018 at p=0 (真正的物理方程式 RHS):
			# dot_rho = ∂_t ρ = ∇²ρ - Pe*∂_y[m(1-ρ)]
			# dot_m   = ∂_t m = ∇²m - Pe*∂_y[ρ(1-ρ)] - 2m
			dot_rho = rho_dd - Pe * grad_fm
			dot_m   = m_dd   - Pe * grad_fr - 2.0 * m_2d
			# relaxation step: explicit Euler (用加號推進時間)
			rho_2d = rho_2d + dt_relax * dot_rho
			m_2d   = m_2d   + dt_relax * dot_m
			# numeric mass correction
			rho_2d = rho_2d - np.mean(rho_2d) + target_mass
			# ====================================================
			# 每 1000 步檢查一次收斂
			# ====================================================
			if step % 1000 == 0:
				max_resid_rho = np.max(np.abs(dot_rho))
				max_resid_m   = np.max(np.abs(dot_m))
				print(f"   [Step {step:5d}] |dot(rho)|_inf = {max_resid_rho:.2e} |dot(m)|_inf = {max_resid_m:.2e}")
				if max_resid_rho < tol and max_resid_m < tol:
					print(f"✅ 成功！第 {step} 步抵達 KH2018 物理穩態！")
					break
		if step == max_steps:
			print("⚠️ 警告：達到最大步數但尚未完全收斂，請考慮增加 max_steps 或檢查初始物理設定。")
		return rho_2d + 0j, m_2d + 0j
	rho1_unrelaxed = rho1.copy()
	rho2_unrelaxed = rho2.copy()
	m1_unrelaxed = m1.copy()
	m2_unrelaxed = m2.copy()
	print("--- 放鬆起點 (rho1, m1) ---")
	rho1, m1 = relax_kh_physical_states(rho1, m1, max_steps=5000000)
	print("--- 放鬆終點 (rho2, m2) ---")
	rho2, m2 = relax_kh_physical_states(rho2, m2, max_steps=5000000)
	np.savez('KHcheckpoints/relaxed.npz', rho1=rho1, rho2=rho2, m1=m1, m2=m2)

	fig_states, axs = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
	mid_x = Lx // 2
	im_u1 = axs[0, 0].imshow(rho1_unrelaxed.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
	axs[0, 0].set_title("Unrelaxed State (rho1)")
	axs[0, 0].set_xlabel("x index")
	axs[0, 0].set_ylabel("y index")
	fig_states.colorbar(im_u1, ax=axs[0, 0], shrink=0.8, label=r"$\rho$")
	# Panel [0, 1]: Unrelaxed rho2
	im_u2 = axs[0, 1].imshow(rho2_unrelaxed.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
	axs[0, 1].set_title("Unrelaxed State (rho2)")
	axs[0, 1].set_xlabel("x index")
	axs[0, 1].set_ylabel("y index")
	fig_states.colorbar(im_u2, ax=axs[0, 1], shrink=0.8, label=r"$\rho$")

	# Panel [0, 2]: Unrelaxed 1D Cross-section
	axs[0, 2].plot(rho1_unrelaxed.real[:, mid_x], 'o-', label='rho1', color='blue', markersize=4)
	axs[0, 2].plot(rho2_unrelaxed.real[:, mid_x], 's-', label='rho2', color='red', markersize=4)
	axs[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
	axs[0, 2].axhline(1, color='gray', linestyle=':', alpha=0.5)
	axs[0, 2].axhline(-1, color='gray', linestyle=':', alpha=0.5)
	axs[0, 2].set_title(f"Unrelaxed 1D Cross-section (at x={mid_x})")
	axs[0, 2].set_ylim(0, 1.5)
	axs[0, 2].set_xlabel("y index")
	axs[0, 2].set_ylabel(r"Density $\rho$")
	axs[0, 2].legend()

	# ------------------------------------------
	# 第二排：已放鬆 (Relaxed)
	# ------------------------------------------
	# Panel [1, 0]: Relaxed rho1
	im_r1 = axs[1, 0].imshow(rho1.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
	axs[1, 0].set_title("Relaxed State (rho1)")
	axs[1, 0].set_xlabel("x index")
	axs[1, 0].set_ylabel("y index")
	fig_states.colorbar(im_r1, ax=axs[1, 0], shrink=0.8, label=r"$\rho$")

	# Panel [1, 1]: Relaxed rho2
	im_r2 = axs[1, 1].imshow(rho2.real, cmap='bwr', vmin=0, vmax=1, origin='lower', aspect='auto')
	axs[1, 1].set_title("Relaxed State (rho2)")
	axs[1, 1].set_xlabel("x index")
	axs[1, 1].set_ylabel("y index")
	fig_states.colorbar(im_r2, ax=axs[1, 1], shrink=0.8, label=r"$\rho$")

	# Panel [1, 2]: Relaxed 1D Cross-section
	axs[1, 2].plot(rho1.real[:, mid_x], 'o-', label='rho1', color='blue', markersize=4)
	axs[1, 2].plot(rho2.real[:, mid_x], 's-', label='rho2', color='red', markersize=4)
	axs[1, 2].set_title(f"Relaxed 1D Cross-section (at x={mid_x})")
	axs[1, 2].set_ylim(0, 1.5)
	axs[1, 2].set_xlabel("y index")
	axs[1, 2].set_ylabel(r"Density $\rho$")
	axs[1, 2].legend()
	fig_states.suptitle("Pe="+str(Pe)+', rho_0='+str(rho_0), fontsize = 20)
	relax_plot_name = f'relaxed_states_Ly{Ly}_Lx{Lx}_Pe{Pe}.png'
	plt.savefig(relax_plot_name, dpi=150, facecolor='white')
	plt.close(fig_states)
	print(f"[*] Saved relaxed states plot to: {relax_plot_name}")
	# ==========================================

rho1k = sp_fft.fft2(rho1)
rho2k = sp_fft.fft2(rho2)
m1k = sp_fft.fft2(m1)
m2k = sp_fft.fft2(m2)

######  INITIAL CONDITIONS 


rho[0,:,:]       = rho1
rho[Ncopy-1,:,:] = rho2
# deterministically set the initial condition

m[0,:,:]       = m1
m[Ncopy-1,:,:] = m2

amp = 0.4
y_coords = np.arange(Ly)
x_coords = np.arange(Lx)
Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

# randomly set the initial condition
noise_amp = 0.5  # noise amplitude
np.random.seed(42 if Lx == 1 else None)

for j in range(1,Ncopy-1):
	tt = float(j)/Ncopy
	linear_rho = rho1*(1-tt) + tt*rho2
	rho[j,:,:] = linear_rho
	linear_m = m1*(1-tt) + tt*m2
	m[j,:,:] = linear_m

# ---- momenta initial guess ----
# deterministic limit in KH note requires p_rho=p_m=0
p_rho[:] = 0.0 + 0j
p_m[:]   = 0.0 + 0j

if os.path.exists(resume_file):
	data = np.load(resume_file)
	rho = data['rho']
	m = data['m']
	p_rho = data['p_rho']
	p_m = data['p_m']
	iteration = data['iteration']
	Lx = data['Lx']
	Ly = data['Ly']
	h = data['h']
	Ncopy = data['Ncopy']
	Tmax = data['Tmax']
	Pe = data['Pe']
	dtau = data['dtau']
	upward = data['upward']
	end_iterations = data['iterations']
	plotStep = data['plotStep']
	start_iter = end_iterations + 1
else:
	start_iter = 0
U_rho = rho + p_rho
U_m = m + p_m	
V_rho = rho - p_rho
V_m = m - p_m
###### Evolve Loop (GDA Algorithm 1: path-time upwind + full reaction) ######
start_time = time.time()
print("start_iter", start_iter)
print("iterations", iterations)
for i in range(start_iter, iterations+1):

	# ================= UPDATE U =================
	# 1. 完整 reaction（含空間擴散 D*Lap）
	# lap_rho   = apply_lap_2d(rho)
	# lap_theta = apply_lap_2d(theta)
	rho_prime = apply_grad_y(rho)
	m_prime   = apply_grad_y(m)
	pr_prime  = apply_grad_y(p_rho)
	pm_prime  = apply_grad_y(p_m)
	lap_pr = apply_lap_2d(p_rho)  # = ∂_y^2 p_rho (因 Lx=1)
	lap_pm = apply_lap_2d(p_m)    # = ∂_y^2 p_m

	# C11 = rho * (1.0 - rho)
	# C12 = m   * (1.0 - rho)
	# C22 = C11

	# # δH/δp_rho = -∂y[bracket ]
	# bracket_1 = Pe * m * (1.0 - rho) - rho_prime + 2.0 * (C11 * pr_prime + C12 * pm_prime)
	# dH_dprho = -apply_grad_y(bracket_1)

	# # δH/δp_m = -∂y[ bracket 2 ] -2m + 2 rho p_m
	# bracket_2 = Pe * rho * (1.0 - rho) - m_prime + 2.0 * (C12 * pr_prime + C22 * pm_prime)
	# dH_dpm = -apply_grad_y(bracket_2) - 2.0*m + 2.0*rho*p_m

	# # δH/δρ
	# dH_drho = (lap_pr
	# 		- Pe*m*pr_prime
	# 		+ Pe*(1.0-2.0*rho)*pm_prime
	# 		+ (1.0-2.0*rho)*(pr_prime**2 + pm_prime**2)
	# 		- 2.0*m*pr_prime*pm_prime
	# 		+ (p_m**2))

	# # δH/δm
	# dH_dm = (lap_pm
	# 		+ Pe*(1.0-rho)*pr_prime
	# 		- 2.0*p_m
	# 		+ 2.0*(1.0-rho)*pr_prime*pm_prime)
	# =========================================================
	# 🛡️ 物理嚴謹防護：係數飽和法 (不會改變真實 rho，質量 100% 守恆)
	# =========================================================
	# 建立一個安全的替身，用來計算物理係數，避免出現負的擴散率
	rho_safe = np.clip(rho.real, 1e-5, 1.0 - 1e-5) + 0j
	m_safe   = np.clip(m.real, -rho_safe.real + 1e-5, rho_safe.real - 1e-5) + 0j

	# 所有的非線性係數 (C11, C12, 以及對流項) 都使用「安全替身」來計算
	C11 = rho_safe * (1.0 - rho_safe)
	C12 = m_safe   * (1.0 - rho_safe)
	C22 = C11

	# δH/δp_rho
	bracket_1 = Pe * m_safe * (1.0 - rho_safe) - rho_prime + 2.0 * (C11 * pr_prime + C12 * pm_prime)
	dH_dprho = -apply_grad_y(bracket_1)

	# δH/δp_m
	bracket_2 = Pe * rho_safe * (1.0 - rho_safe) - m_prime + 2.0 * (C12 * pr_prime + C22 * pm_prime)
	dH_dpm = -apply_grad_y(bracket_2) - 2.0*m + 2.0*rho_safe*p_m

	# δH/δρ
	dH_drho = (lap_pr
			- Pe * m_safe * pr_prime
			+ Pe * (1.0 - 2.0 * rho_safe) * pm_prime
			+ (1.0 - 2.0 * rho_safe) * (pr_prime**2 + pm_prime**2)
			- 2.0 * m_safe * pr_prime * pm_prime
			+ (p_m**2))

	# δH/δm
	dH_dm = (lap_pm
			+ Pe * (1.0 - rho_safe) * pr_prime
			- 2.0 * p_m
			+ 2.0 * (1.0 - rho_safe) * pr_prime * pm_prime)

	# reaction_U = dH_drho - dH_dtheta
	reaction_U_rho = dH_drho - dH_dprho
	reaction_U_m   = dH_dm - dH_dpm

	# 2. 空間 FFT2
	U_rho_Fourier[:] = sp_fft.fft2(U_rho, axes=(1, 2))
	U_m_Fourier[:] = sp_fft.fft2(U_m, axes=(1, 2))
	V_rho_Fourier[:] = sp_fft.fft2(V_rho, axes=(1, 2))
	V_m_Fourier[:] = sp_fft.fft2(V_m, axes=(1, 2))
	reaction_U_rho_Fourier[:] = sp_fft.fft2(reaction_U_rho, axes=(1, 2))
	reaction_U_m_Fourier[:] = sp_fft.fft2(reaction_U_m, axes=(1, 2))

	RHS_U_rho_Fourier = U_rho_Fourier + dtau * reaction_U_rho_Fourier
	RHS_U_rho_Fourier[Ncopy-1, :, :] = -V_rho_Fourier[Ncopy-1, :, :] + 2.0 * rho2k
	RHS_U_m_Fourier = U_m_Fourier + dtau * reaction_U_m_Fourier
	RHS_U_m_Fourier[Ncopy-1, :, :] = -V_m_Fourier[Ncopy-1, :, :] + 2.0 * m2k
	# RHS_flat = RHS_U_Fourier.reshape(Ncopy, -1)
	RHS_U_rho_flat = RHS_U_rho_Fourier.reshape(Ncopy, -1)
	RHS_U_m_flat = RHS_U_m_Fourier.reshape(Ncopy, -1)
	# U_Fourier_flat = U_Fourier.reshape(Ncopy, -1)
	U_rho_Fourier_flat = U_rho_Fourier.reshape(Ncopy, -1)
	U_m_Fourier_flat = U_m_Fourier.reshape(Ncopy, -1)
	# U_new_flat = np.zeros_like(RHS_flat)
	U_rho_new_flat = np.zeros_like(RHS_U_rho_flat)
	U_m_new_flat = np.zeros_like(RHS_U_m_flat)
	for col in range(Ly):
		stab_term = gamma * (2.0 * k2_flat[col]) * U_rho_Fourier_flat[:, col]
		RHS_col = RHS_U_rho_flat[:, col] + dtau * stab_term
		RHS_col[-1] = RHS_U_rho_flat[-1, col]
		U_rho_new_flat[:, col] = solve_banded((0, 2), A_banded[col], RHS_col)
		stab_term = gamma * (2.0 * k2_flat[col]) * U_m_Fourier_flat[:, col]
		RHS_col = RHS_U_m_flat[:, col] + dtau * stab_term
		RHS_col[-1] = RHS_U_m_flat[-1, col]
		U_m_new_flat[:, col] = solve_banded((0, 2), A_banded[col], RHS_col)
	# U2_Fourier[:] = U_new_flat.reshape(Ncopy, Ly, Lx)
	U_rho_Fourier[:] = U_rho_new_flat.reshape(Ncopy, Ly, Lx)
	U_m_Fourier[:] = U_m_new_flat.reshape(Ncopy, Ly, Lx)

	U_rho[:] = sp_fft.ifft2(U_rho_Fourier, axes=(1, 2)).real
	U_m[:] = sp_fft.ifft2(U_m_Fourier, axes=(1, 2)).real
	rho   = 0.5 * (U_rho + V_rho)
	p_rho = 0.5 * (U_rho - V_rho)
	m     = 0.5 * (U_m + V_m)
	p_m   = 0.5 * (U_m - V_m)
	rho[0] = rho1
	m[0] = m1
	rho[Ncopy-1] = rho2
	m[Ncopy-1] = m2
	p_rho[0] = 0.0 + 0j
	p_m[0] = 0.0 + 0j
	p_rho[Ncopy-1] = 0.0 + 0j
	p_m[Ncopy-1] = 0.0 + 0j
	U_rho = rho + p_rho
	U_m = m + p_m
	V_rho = rho - p_rho
	V_m = m - p_m

	# ================= UPDATE V =================
	rho_prime = apply_grad_y(rho)
	m_prime   = apply_grad_y(m)
	pr_prime  = apply_grad_y(p_rho)
	pm_prime  = apply_grad_y(p_m)
	lap_pr = apply_lap_2d(p_rho)  # = ∂_y^2 p_rho (因 Lx=1)
	lap_pm = apply_lap_2d(p_m)    # = ∂_y^2 p_m
	# C11 = rho * (1.0 - rho)
	# C12 = m   * (1.0 - rho)
	# C22 = C11

	# # δH/δp_rho = -∂y[bracket 1 ]
	# bracket_1 = Pe * m * (1.0 - rho) - rho_prime + 2.0 * (C11 * pr_prime + C12 * pm_prime)
	# dH_dprho = -apply_grad_y(bracket_1)

	# # δH/δp_m = -∂y[ bracket 2 ] -2m + 2 rho p_m
	# bracket_2 = Pe * rho * (1.0 - rho) - m_prime + 2.0 * (C12 * pr_prime + C22 * pm_prime)
	# dH_dpm = -apply_grad_y(bracket_2) - 2.0*m + 2.0*rho*p_m

	# # δH/δρ
	# dH_drho = (lap_pr
	# 		- Pe*m*pr_prime
	# 		+ Pe*(1.0-2.0*rho)*pm_prime
	# 		+ (1.0-2.0*rho)*(pr_prime**2 + pm_prime**2)
	# 		- 2.0*m*pr_prime*pm_prime
	# 		+ (p_m**2))

	# # δH/δm
	# dH_dm = (lap_pm
	# 		+ Pe*(1.0-rho)*pr_prime
	# 		- 2.0*p_m
	# 		+ 2.0*(1.0-rho)*pr_prime*pm_prime)
	# =========================================================
	# 🛡️ 物理嚴謹防護：係數飽和法 (不會改變真實 rho，質量 100% 守恆)
	# =========================================================
	# 建立一個安全的替身，用來計算物理係數，避免出現負的擴散率
	rho_safe = np.clip(rho.real, 1e-5, 1.0 - 1e-5) + 0j
	m_safe   = np.clip(m.real, -rho_safe.real + 1e-5, rho_safe.real - 1e-5) + 0j

	# 所有的非線性係數 (C11, C12, 以及對流項) 都使用「安全替身」來計算
	C11 = rho_safe * (1.0 - rho_safe)
	C12 = m_safe   * (1.0 - rho_safe)
	C22 = C11

	# δH/δp_rho
	bracket_1 = Pe * m_safe * (1.0 - rho_safe) - rho_prime + 2.0 * (C11 * pr_prime + C12 * pm_prime)
	dH_dprho = -apply_grad_y(bracket_1)

	# δH/δp_m
	bracket_2 = Pe * rho_safe * (1.0 - rho_safe) - m_prime + 2.0 * (C12 * pr_prime + C22 * pm_prime)
	dH_dpm = -apply_grad_y(bracket_2) - 2.0*m + 2.0*rho_safe*p_m

	# δH/δρ
	dH_drho = (lap_pr
			- Pe * m_safe * pr_prime
			+ Pe * (1.0 - 2.0 * rho_safe) * pm_prime
			+ (1.0 - 2.0 * rho_safe) * (pr_prime**2 + pm_prime**2)
			- 2.0 * m_safe * pr_prime * pm_prime
			+ (p_m**2))

	# δH/δm
	dH_dm = (lap_pm
			+ Pe * (1.0 - rho_safe) * pr_prime
			- 2.0 * p_m
			+ 2.0 * (1.0 - rho_safe) * pr_prime * pm_prime)
	
	reaction_V_rho = dH_drho + dH_dprho
	reaction_V_m   = dH_dm + dH_dpm
	reaction_V_rho_Fourier[:] = sp_fft.fft2(reaction_V_rho, axes=(1, 2))
	reaction_V_m_Fourier[:] = sp_fft.fft2(reaction_V_m, axes=(1, 2))
	U_rho_Fourier[:] = sp_fft.fft2(U_rho, axes=(1, 2))
	U_m_Fourier[:] = sp_fft.fft2(U_m, axes=(1, 2))
	V_rho_Fourier[:] = sp_fft.fft2(V_rho, axes=(1, 2))
	V_m_Fourier[:] = sp_fft.fft2(V_m, axes=(1, 2))

	RHS_V_rho_Fourier = V_rho_Fourier + dtau * reaction_V_rho_Fourier
	RHS_V_rho_Fourier[0, :, :] = -U_rho_Fourier[0, :, :] + 2.0 * rho1k
	RHS_V_m_Fourier = V_m_Fourier + dtau * reaction_V_m_Fourier
	RHS_V_m_Fourier[0, :, :] = -U_m_Fourier[0, :, :] + 2.0 * m1k
	

	RHS_V_rho_flat = RHS_V_rho_Fourier.reshape(Ncopy, -1)
	RHS_V_m_flat = RHS_V_m_Fourier.reshape(Ncopy, -1)
	V_rho_Fourier_flat = V_rho_Fourier.reshape(Ncopy, -1)
	V_m_Fourier_flat = V_m_Fourier.reshape(Ncopy, -1)
	V_rho_new_flat = np.zeros_like(RHS_V_rho_flat)
	V_m_new_flat = np.zeros_like(RHS_V_m_flat)
	for col in range(Ly):
		stab_term = gamma * (2.0 * k2_flat[col]) * V_rho_Fourier_flat[:, col]
		RHS_col = RHS_V_rho_flat[:, col] + dtau * stab_term
		RHS_col[0] = RHS_V_rho_flat[0, col]
		V_rho_new_flat[:, col] = solve_banded((2, 0), B_banded[col], RHS_col)
		stab_term = gamma * (2.0 * k2_flat[col]) * V_m_Fourier_flat[:, col]
		RHS_col = RHS_V_m_flat[:, col] + dtau * stab_term
		RHS_col[0] = RHS_V_m_flat[0, col]
		V_m_new_flat[:, col] = solve_banded((2, 0), B_banded[col], RHS_col)
	V_rho_Fourier[:] = V_rho_new_flat.reshape(Ncopy, Ly, Lx)
	V_m_Fourier[:] = V_m_new_flat.reshape(Ncopy, Ly, Lx)
	V_rho[:] = sp_fft.ifft2(V_rho_Fourier, axes=(1, 2)).real
	V_m[:] = sp_fft.ifft2(V_m_Fourier, axes=(1, 2)).real
	rho   = 0.5 * (U_rho + V_rho)
	p_rho = 0.5 * (U_rho - V_rho)
	m     = 0.5 * (U_m + V_m)
	p_m   = 0.5 * (U_m - V_m)
	rho[0] = rho1
	m[0] = m1
	rho[Ncopy-1] = rho2
	m[Ncopy-1] = m2
	p_rho[0] = 0.0 + 0j
	p_m[0] = 0.0 + 0j
	p_rho[Ncopy-1] = 0.0 + 0j
	p_m[Ncopy-1] = 0.0 + 0j
	# U = rho + theta
	# V = rho - theta
	U_rho = rho + p_rho
	U_m = m + p_m
	V_rho = rho - p_rho
	V_m = m - p_m


	if i > start_iter:
		# 指標一：數值穩定度 (路徑是否不再移動)
		max_diff_rho = np.max(np.abs(rho - old_rho_for_check))
		max_diff_m = np.max(np.abs(m - old_m_for_check))
		max_diff_pr = np.max(np.abs(p_rho - old_pr_for_check))
		max_diff_pm = np.max(np.abs(p_m - old_pm_for_check))
		# 指標二：物理精確度 (Hamiltonian 是否平坦)	
		# 1. 取得壓平的 rho 和 theta 來算整條路徑的 H
		rho_1d_check = rho.reshape(Ncopy, -1)
		m_1d_check = m.reshape(Ncopy, -1)
		# theta_1d_check = theta.reshape(Ncopy, -1)
		pr_1d_check = p_rho.reshape(Ncopy, -1)
		pm_1d_check = p_m.reshape(Ncopy, -1)

		H_path = Hamiltonian_KH_1D(Pe, rho_1d_check, m_1d_check, pr_1d_check, pm_1d_check).real

		# 2. 計算 H 的標準差 (跳過頭尾兩點，避免邊界差分帶來的微小數值震盪)
		H_std = np.std(H_path[2:-10])

		# 判定條件：路徑幾乎不動 ( < 1e-5 ) 且 H 非常平坦 ( 標準差 < 1e-3 )
		if max_diff_rho < 5e-5 and H_std < Hamiltonian_KH_1D(Pe, rho_1d_check, m_1d_check, pr_1d_check, pm_1d_check).real[2:-10].max()*1e-2:
			print(f"\n✨ 恭喜！系統在第 {i} 步已完美收斂抵達 Instanton 鞍點！")
			print(f"最大數值誤差 dRho: {max_diff_rho:.2e} | 最大數值誤差 dM: {max_diff_m:.2e} | 物理守恆 H 標準差: {H_std:.2e}")
			break  # 安全跳出，完美存檔！
			
	# 紀錄當前狀態，留給下一步比對
	old_rho_for_check = rho.copy()
	old_m_for_check = m.copy()
	old_pr_for_check = p_rho.copy()
	old_pm_for_check = p_m.copy()
	##################################################
	'''------------ THIRD: PLOT DATA ---------'''
	##################################################	
	    
	if( i%plotStep == 0):
		if i > start_iter:
			print('max_diff_rho, max_diff_m, max_diff_pr, max_diff_pm, H_std, H_max:', max_diff_rho, max_diff_m, max_diff_pr, max_diff_pm, H_std, Hamiltonian_KH_1D(Pe, rho_1d_check, m_1d_check, pr_1d_check, pm_1d_check).real[2:-10].max()*1e-2)
		# Flatten space to (Ncopy, Ly*Lx) for Lagrangian/Hamiltonian and plots
		rho_1d = rho.reshape(Ncopy, -1)
		m_1d = m.reshape(Ncopy, -1)
		pr_1d = p_rho.reshape(Ncopy, -1)
		pm_1d = p_m.reshape(Ncopy, -1)
		plt.gcf()
		fig = plt.figure(figsize=(10,10),layout='constrained')
		# 4 subplots: 1. rho, 2. theta, 3. Lagrangian, 4. Hamiltonian 2 rows 2 columns
		ax0 = fig.add_subplot(325)
		ax1 = fig.add_subplot(321)
		ax2 = fig.add_subplot(322)
		ax3 = fig.add_subplot(323)
		ax4 = fig.add_subplot(324)
		ax5 = fig.add_subplot(326)
		Lag = Lagrangian_KH_1D(Pe, ds, rho_1d, m_1d, pr_1d, pm_1d).real
		actionS = dnu* np.sum(Lag)
		
		fig.suptitle(r'$N_\mathrm{copy}=$'+str(int(Ncopy))+r', $L=$'+str(Ly)+r', $\Delta \tau=$'+str("%.1e"%dtau)+r', $Pe=$'+str("%.1e"%Pe) +r', $T_\mathrm{max}=$'+str("%.1f"%Tmax)+r', $S=$'+str("%.6f"%(actionS))+', Time '+ str(i*dtau), fontsize=20)
		
		### [NEW] PLOT: Symmetry Breaking Projection (Mean vs Std)
		mean_rho = np.mean(rho_1d.real, axis=1) 
		mean_m = np.mean(m_1d.real, axis=1)
		std_rho  = np.std(rho_1d.real, axis=1)  	
		std_m = np.std(m_1d.real, axis=1)

		im1 = ax0.scatter(std_rho, mean_rho, c=np.linspace(0,1,Ncopy), cmap='viridis', s=15, zorder=10) 
		im2 = ax0.scatter(std_m, mean_m, c=np.linspace(0,1,Ncopy), cmap='viridis', s=15, zorder=10) 
		ax0.plot(std_rho, mean_rho, color='darkblue', linewidth=2, label='Instanton Path')
		ax0.plot(std_m, mean_m, color='darkred', linewidth=2, label='Instanton Path')
		ax0.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Homogeneous (Symmetric)')

		ax0.set_xlabel(r'Inhomogeneity $\sigma_\rho$ (Std Dev)', fontsize=15)
		ax0.set_ylabel(r'Mean Density $\bar{\rho}$', fontsize=15)
		ax0.set_title(r'Symmetry Breaking Projection', fontsize=16)
        
		ax0.grid(True, linestyle=':', alpha=0.6)
		ax0.set_xlim(left=-0.05, right=max(std_rho.max()*1.2, 0.5)) 
		ax0.set_ylim(0, 1.5)
		ax0.legend(loc='best', fontsize=12)
		## colorbar
		cbar1 = fig.colorbar(im1, ax=ax0, shrink=0.8)
		cbar2 = fig.colorbar(im2, ax=ax0, shrink=0.8)
		cbar1.set_label(r'$\sigma_\rho$', fontsize=20)
		cbar2.set_label(r'$\sigma_m$', fontsize=20)

		### PLOT rho
		mid_x = Lx // 2
		rho_slice = rho[:, :, mid_x].real.T
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		y_edges = np.arange(Ly + 1) - 0.5 # 現在 Y 軸是 X 空間座標
		im = ax1.pcolormesh(t_edges, y_edges, rho_slice, 
                           cmap='bwr', 
                           shading='flat',
                           vmin=0, vmax=1)
		cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
		cbar.set_label(r'$\rho(x_{mid}, y, t)$', fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=15)
		ax1.set_yticks([0, Ly-1])
		ax1.set_xlabel('Time', fontsize=20)
		ax1.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)
		Lag = Lagrangian_KH_1D(Pe, ds, rho_1d, m_1d, pr_1d, pm_1d).real
		actionS = dnu* np.sum(Lag)

		### PLOT m
		mid_x = Lx // 2
		m_slice = m[:, :, mid_x].real.T
		t_edges = np.linspace(0, Tmax, Ncopy + 1)
		y_edges = np.arange(Ly + 1) - 0.5 # 現在 Y 軸是 X 空間座標
		im2 = ax2.pcolormesh(t_edges, y_edges, m_slice, 
                           cmap='bwr', 
                           shading='flat',
                           vmin=-1.5, vmax=1.5)
		cbar = fig.colorbar(im2, ax=ax2, shrink=0.8)
		cbar.set_label(r'$m(x_{mid}, y, t)$', fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=15)
		ax2.set_yticks([0, Ly-1])
		ax2.set_xlabel('Time', fontsize=20)
		ax2.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)
		Lag = Lagrangian_KH_1D(Pe, ds, rho_1d, m_1d, pr_1d, pm_1d).real
		actionS = dnu* np.sum(Lag)

		### PLOT p_rho
		p_rho_slice = p_rho[:, :, mid_x].real.T
		im3 = ax3.pcolormesh(t_edges, y_edges, p_rho_slice, 
							cmap='bwr', 
							shading='flat',
							vmin=min(p_rho_slice.min(), -0.5), vmax=max(p_rho_slice.max(), 0.5))
		cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.8)
		cbar3.set_label(r'$p_\rho(x_{mid}, y, t)$', fontsize=20)
		ax3.tick_params(axis='both', which='major', labelsize=15)
		ax3.set_yticks([0, Ly-1])
		ax3.set_xlabel('Time', fontsize=20)
		ax3.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)

		### PLOT p_m
		p_m_slice = p_m[:, :, mid_x].real.T
		im4 = ax4.pcolormesh(t_edges, y_edges, p_m_slice, 
							cmap='bwr', 
							shading='flat',
							vmin=min(p_m_slice.min(), -0.5), vmax=max(p_m_slice.max(), 0.5))
		cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.8)
		cbar4.set_label(r'$p_m(x_{mid}, y, t)$', fontsize=20)
		ax4.tick_params(axis='both', which='major', labelsize=15)
		ax4.set_yticks([0, Ly-1])
		ax4.set_xlabel('Time', fontsize=20)
		ax4.set_ylabel(f'Y Coordinate (at x={mid_x})', fontsize=15)

		### PLOT Lagrangian and Hamiltonian
		ax5.set_aspect('auto')
		ax5.plot(np.linspace(0,1,Ncopy), Lag, label=r'$L(\rho,\dot\rho)$', color='black'  ) 
		ax5.plot(np.linspace(0,1,Ncopy), Hamiltonian_KH_1D(Pe, rho_1d, m_1d, pr_1d, pm_1d).real, label=r'$H(\rho,m,\dot\rho,\dot m)$', color='brown' , linestyle='-.') 
		plt.legend(loc='best', fontsize=14)
		ax5.tick_params(axis='both', which='major', labelsize=15)
		ax5.set_xlabel(r'$t/T_\mathrm{Max}$', fontsize=20)
		ax5.set_ylabel(r'$L$', fontsize=20)
		ax5.set_xlim(0,1)
		ax5.set_ylim(-0.01*max(Lag[2:-10].max(), Hamiltonian_KH_1D(Pe, rho_1d, m_1d, pr_1d, pm_1d).real[2:-10].max()), max(Lag[2:-10].max(), Hamiltonian_KH_1D(Pe, rho_1d, m_1d, pr_1d, pm_1d).real[2:-10].max())*1.2)
		
		if(upward==True):
			plt.savefig('upward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'Pe'+str(Pe)+'_rho_0'+str(rho_0)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		else:
			plt.savefig('downward_Lx'+str(int(Lx))+'_Ly'+str(int(Ly))+'_N'+str(int(Ncopy))+'h'+str(float(h))+'Pe'+str(Pe)+'_rho_0'+str(rho_0)+'_dtau'+str("%.1e"%dtau)+'_'+"%09d" % (i,)+'.png', format='png', bbox_inches='tight')
			print(f"Time taken: {time.time() - start_time} seconds", "iteration", i)
		plt.clf()
		plt.close()
		np.savez_compressed('KHcheckpoints/checkpoint.npz', rho=rho, m=m, p_rho=p_rho, p_m=p_m, iteration=i, Lx=Lx, Ly=Ly, h=h, Ncopy=Ncopy, Tmax=Tmax, Pe=Pe, dtau=dtau, upward=upward, iterations=i, plotStep=plotStep)
def animate_2d_heatmap(rho, m, p_rho, p_m, filename="2d_evolution.mp4", dnu=None, skip=1, fps=30):
	"""
	2D Heatmap Animation for both rho and m
	rho, theta shape: (Ncopy, Ly, Lx)
	"""
	rho = np.asarray(rho).real
	m = np.asarray(m).real
	p_rho = np.asarray(p_rho).real
	p_m = np.asarray(p_m).real
	Nt, Ly, Lx = rho.shape

	if dnu is None:
		dnu = 1.0 / max(Nt - 1, 1)

	# Downsample frames
	indices = np.arange(0, Nt, skip)
	if len(indices) > 300: # 限制最大幀數以免檔案太大
		indices = np.linspace(0, Nt - 1, 200, dtype=int)

	rho_sub = rho[indices]
	m_sub = m[indices]
	p_rho_sub = p_rho[indices]
	p_m_sub = p_m[indices]
	n_frames = len(indices)

	# 既然是曲線圖，4 列 1 欄 (4 rows, 1 column) 是最適合觀察介面波動的佈局
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), layout='constrained')

	# 準備 X 軸數據（即 y 方向的座標）
	y_points = np.arange(Ly)

	# ---------------------------------------------------------
	# 1. 初始化 Rho 曲線
	# ---------------------------------------------------------
	# rho_sub[0] 的形狀是 (Ly, 1)，我們取 [:, 0] 轉成 1D 向量
	line1, = ax1.plot(y_points, rho_sub[0].real[:, 0], color='red', lw=2)
	ax1.set_ylim(-0.1, 1.1)  # Rho 通常在 0~1 之間
	ax1.set_title(r"Density $\rho$")
	ax1.set_ylabel(r"Value")
	ax1.grid(True, alpha=0.3)

	# ---------------------------------------------------------
	# 2. 初始化 M 曲線
	# ---------------------------------------------------------
	m_max = max(np.abs(m).max(), 0.01) * 1.2 # 給一點邊界緩衝
	line2, = ax2.plot(y_points, m_sub[0].real[:, 0], color='blue', lw=2)
	ax2.set_ylim(-m_max, m_max)
	ax2.set_title(r"Momentum $m$")
	ax2.set_ylabel(r"Value")
	ax2.grid(True, alpha=0.3)

	# ---------------------------------------------------------
	# 3. 初始化 P_rho 曲線
	# ---------------------------------------------------------
	p_rho_max = max(np.abs(p_rho).max(), 0.01) * 1.2
	line3, = ax3.plot(y_points, p_rho_sub[0].real[:, 0], color='green', lw=2)
	ax3.set_ylim(-p_rho_max, p_rho_max)
	ax3.set_title(r"Momentum $p_\rho$")
	ax3.set_ylabel(r"Value")
	ax3.grid(True, alpha=0.3)

	# ---------------------------------------------------------
	# 4. 初始化 P_m 曲線
	# ---------------------------------------------------------
	p_m_max = max(np.abs(p_m).max(), 0.01) * 1.2
	line4, = ax4.plot(y_points, p_m_sub[0].real[:, 0], color='purple', lw=2)
	ax4.set_ylim(-p_m_max, p_m_max)
	ax4.set_title(r"Momentum $p_m$")
	ax4.set_ylabel(r"Value")
	ax4.set_xlabel("y coordinate")
	ax4.grid(True, alpha=0.3)

	# 共同大標題
	fig.suptitle(f"Time: 0.00 | Profile Evolution", fontsize=16)
	def update(frame_idx):
		# 更新各條曲線的數據
		line1.set_ydata(rho_sub[frame_idx].real[:, 0])
		line2.set_ydata(m_sub[frame_idx].real[:, 0])
		line3.set_ydata(p_rho_sub[frame_idx].real[:, 0])
		line4.set_ydata(p_m_sub[frame_idx].real[:, 0])
		
		t = indices[frame_idx] * dnu
		fig.suptitle(f"Time: {t:.2f} | Profile Evolution")
		
		return line1, line2, line3, line4

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
animate_2d_heatmap(rho, m, p_rho, p_m, filename=f"KH_2d_sim_Lx{int(Lx)}_Ly{int(Ly)}.mp4", dnu=dnu, skip=4, fps=10)












